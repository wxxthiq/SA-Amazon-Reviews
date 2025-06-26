import streamlit as st
import pandas as pd
import sqlite3
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import altair as alt
import re
from collections import Counter
import spacy
from textblob import TextBlob
import os
import requests
import zipfile
import subprocess
import json
import plotly.express as px


# --- Altair Data Transformer ---
alt.data_transformers.enable('default', max_rows=None)

# --- Configuration ---
KAGGLE_USERNAME = "wathiqsoualhi"
KAGGLE_DATASET_SLUG = "mcauley-v2"
DATA_VERSION = 0 # Increment this number to force a re-download of your data
DATABASE_PATH = "amazon_reviews_images.db" 
VERSION_FILE_PATH = ".db_version"
PRODUCTS_PER_PAGE = 16
PLACEHOLDER_IMAGE_URL = "https://via.placeholder.com/150"

def download_data_with_versioning(dataset_slug, db_path, version_path, expected_version):
    """Downloads data only if the local version is out of date."""
    current_version = 0
    if os.path.exists(version_path):
        with open(version_path, "r") as f:
            try: current_version = int(f.read().strip())
            except (ValueError, TypeError): current_version = 0
    
    if current_version == expected_version and os.path.exists(db_path):
        return

    st.info(f"Database v{current_version} is outdated (expected v{expected_version}). Forcing fresh download...")
    if os.path.exists(db_path): os.remove(db_path)
    if os.path.exists(version_path): os.remove(version_path)

    url = f"https://www.kaggle.com/api/v1/datasets/download/{dataset_slug}"
    zip_file_path = f"{db_path}.zip"
    
    with st.spinner(f"Downloading data from Kaggle... Please wait."):
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(zip_file_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192): f.write(chunk)
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                zip_ref.extractall(".")
            os.remove(zip_file_path)
            
            if os.path.exists(db_path):
                with open(version_path, "w") as f: f.write(str(expected_version))
                st.success("Database download complete! Rerunning app...")
                st.rerun()
            else:
                st.error(f"FATAL: Download complete, but '{db_path}' was not found after unzipping. Please check the name in your Kaggle zip file.")
                st.stop()
        except Exception as e:
            st.error(f"FATAL: An error occurred during download: {e}")
            st.stop()

@st.cache_resource
def connect_to_db(path):
    try:
        return sqlite3.connect(path, uri=True, check_same_thread=False, timeout=15)
    except Exception as e:
        st.error(f"FATAL: Could not connect to database. Error: {e}")
        return None

@st.cache_resource
def load_spacy_model():
    model_name = "en_core_web_sm"
    try: nlp = spacy.load(model_name)
    except OSError:
        with st.spinner(f"Downloading spaCy model '{model_name}'..."):
            subprocess.run(["python", "-m", "spacy", "download", model_name], check=True)
            nlp = spacy.load(model_name)
    return nlp

@st.cache_data
def get_all_categories(_conn):
    if _conn is None: return ["All"]
    try:
        df = pd.read_sql_query("SELECT DISTINCT category FROM reviews", _conn)
        categories = sorted(df['category'].unique().tolist())
        categories.insert(0, "All")
        return categories
    except Exception as e:
        st.warning(f"Could not load categories: {e}")
        return ["All"]

@st.cache_data
def get_product_summary_data(_conn):
    if _conn is None: return pd.DataFrame()
    try:
        query = """
        SELECT
            parent_asin, MAX(product_title) as product_title, MAX(image_url) as image_url,
            MAX(category) as category, average_rating, review_count
        FROM reviews
        WHERE parent_asin IS NOT NULL AND product_title IS NOT NULL
        GROUP BY parent_asin, average_rating, review_count
        """
        df = pd.read_sql_query(query, _conn)
        return df
    except Exception as e:
        st.error(f"An error occurred while loading product summary: {e}")
        return pd.DataFrame()

def search_products(products_df, category, search_term, sort_by):
    if products_df is None or products_df.empty: return pd.DataFrame()
    results_df = products_df.copy().dropna(subset=['product_title', 'parent_asin'])
    if category != "All":
        results_df = results_df[results_df['category'] == category]
    if search_term:
        results_df = results_df[results_df['product_title'].str.contains(search_term, case=False, na=False)]
    if sort_by == "Popularity (Most Reviews)":
        results_df = results_df.sort_values(by="review_count", ascending=False, na_position='last')
    elif sort_by == "Highest Rating":
        results_df = results_df.sort_values(by="average_rating", ascending=False, na_position='last')
    elif sort_by == "Lowest Rating":
        results_df = results_df.sort_values(by="average_rating", ascending=True, na_position='first')
    return results_df

def fetch_reviews_for_product(_conn, asin):
    if _conn is None: return pd.DataFrame()
    query = "SELECT * FROM reviews WHERE parent_asin = ?"
    df = pd.read_sql_query(query, _conn, params=(asin,))
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.reset_index().rename(columns={'index': 'review_id'})
    return df

@st.cache_data
def calculate_discrepancy(reviews_df):
    df = reviews_df.copy()
    df.dropna(subset=['text'], inplace=True)
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    df.dropna(subset=['rating'], inplace=True)
    df['normalized_rating'] = (df['rating'] - 3) / 2
    df['text_polarity'] = df['text'].astype(str).apply(lambda x: TextBlob(x).sentiment.polarity)
    df['discrepancy'] = (df['text_polarity'] - df['normalized_rating']).abs()
    df['rating_display'] = df['rating'].round().astype(int)
    return df

@st.cache_data
def aspect_based_sentiment(_reviews_df, aspects, _nlp_model):
    aspect_sentiments = {aspect: {'pos': 0, 'neg': 0, 'neu': 0, 'mentions': 0} for aspect in aspects}
    for review_text in _reviews_df['text'].dropna():
        for sentence in _nlp_model(review_text).sents:
            for aspect in aspects:
                if f' {aspect} ' in sentence.text.lower():
                    aspect_sentiments[aspect]['mentions'] += 1
                    sentiment = TextBlob(sentence.text).sentiment.polarity
                    if sentiment > 0.1: aspect_sentiments[aspect]['pos'] += 1
                    elif sentiment < -0.1: aspect_sentiments[aspect]['neg'] += 1
                    else: aspect_sentiments[aspect]['neu'] += 1
    results = [{'aspect': aspect, **scores} for aspect, scores in aspect_sentiments.items() if scores['mentions'] > 0]
    return pd.DataFrame(results)

def get_top_keywords(text, custom_stopwords, top_n=50):
    if not isinstance(text, str): return []
    stopwords = STOPWORDS.union(set(custom_stopwords))
    words = re.findall(r"[a-zA-Z']+", text.lower())
    filtered_words = [word for word in words if word not in stopwords and len(word) > 2]
    word_counts = Counter(filtered_words)
    return [word for word, count in word_counts.most_common(top_n)]

def generate_wordcloud(text, title, custom_stopwords):
    stopwords = STOPWORDS.union(set(custom_stopwords))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title(title, fontsize=16)
    if text and isinstance(text, str) and text.strip():
        try:
            wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=stopwords, colormap='viridis').generate(text)
            ax.imshow(wordcloud, interpolation='bilinear')
        except ValueError: ax.text(0.5, 0.5, 'No words to display', ha='center', va='center')
    else: ax.text(0.5, 0.5, 'No reviews for this data segment', ha='center', va='center')
    ax.axis('off')
    return fig


# --- Main App Execution ---
st.set_page_config(layout="wide", page_title="Amazon Review Explorer")
st.title("⚡ Amazon Reviews - Sentiment Dashboard")

KAGGLE_DATASET = f"{KAGGLE_USERNAME}/{KAGGLE_DATASET_SLUG}"
download_data_with_versioning(KAGGLE_DATASET, DATABASE_PATH, VERSION_FILE_PATH, DATA_VERSION)

conn = connect_to_db(DATABASE_PATH)
nlp = load_spacy_model()

# Initialize session state variables
if 'page' not in st.session_state: st.session_state.page = 0
if 'selected_product' not in st.session_state: st.session_state.selected_product = None
if 'search_results' not in st.session_state: st.session_state.search_results = pd.DataFrame()


# --- Main App Logic ---
if conn and nlp:
    products_df = get_product_summary_data(conn)
    
    if st.session_state.selected_product:
        # --- DETAILED PRODUCT VIEW ---
        selected_asin = st.session_state.selected_product
        product_details_series = products_df[products_df['parent_asin'] == selected_asin]
        
        if product_details_series.empty:
            st.error(f"Could not find details for product ASIN: {selected_asin}")
            if st.button("⬅️ Back to Search"):
                st.session_state.selected_product = None
                st.rerun()
            st.stop()
        else:
            product_details = product_details_series.iloc[0]

        product_reviews = fetch_reviews_for_product(conn, selected_asin)
        
        if st.button("⬅️ Back to Search Results"):
            st.session_state.selected_product = None
            st.rerun()

        if not product_reviews.empty:
            col1, col2 = st.columns([1, 4])
            with col1:
                # FIX: Use the image_url directly, no helper function needed
                st.image(product_details['image_url'] or PLACEHOLDER_IMAGE_URL, use_container_width=True)
            with col2:
                st.header(product_details['product_title'])
                stat_cols = st.columns(3)
                stat_cols[0].metric("Average Rating", f"{product_details['average_rating']:.2f} ⭐" if pd.notna(product_details['average_rating']) else "N/A")
                stat_cols[1].metric("Total Reviews", int(product_details['review_count']) if pd.notna(product_details['review_count']) else 0)
                stat_cols[2].metric("Category", product_details['category'])
            st.markdown("---")
            
            with st.spinner("Running advanced analysis on reviews..."):
                reviews_with_scores = calculate_discrepancy(product_reviews)
                # Define aspects relevant to fashion
                fashion_aspects = ['fit', 'size', 'color', 'fabric', 'quality', 'price', 'style', 'comfort', 'stitching', 'zipper']
                absa_results_df = aspect_based_sentiment(reviews_with_scores, fashion_aspects, nlp)
            
            overview_tab, keyword_tab = st.tabs(["📊 Comprehensive Analysis", "🔎 Keyword Explorer"])

            with overview_tab:
                st.subheader("Deep Dive into Review Sentiment")
                col1, col2 = st.columns(2, gap="large")
                with col1:
                    st.markdown("#### Aspect Sentiment")
                    st.info("What specific features do customers mention positively or negatively?", icon="💡")
                    if not absa_results_df.empty:
                        absa_melted = absa_results_df.melt(id_vars=['aspect'], value_vars=['pos', 'neg'], var_name='sentiment', value_name='count')
                        absa_melted['count'] = absa_melted.apply(lambda row: -row['count'] if row['sentiment'] == 'neg' else row['count'], axis=1)
                        absa_chart = alt.Chart(absa_melted).mark_bar().encode(x=alt.X('count:Q', title='Number of Mentions'), y=alt.Y('aspect:N', sort='-x', title='Product Aspect'), color=alt.Color('sentiment:N', scale=alt.Scale(domain=['pos', 'neg'], range=['#2ca02c', '#d62728']), legend=alt.Legend(title="Sentiment")), tooltip=[alt.Tooltip('aspect:N'), alt.Tooltip('count:Q', title='Mentions')]).properties(title="Positive vs. Negative Mentions by Feature")
                        st.altair_chart(absa_chart, use_container_width=True)
                    else: st.warning("No defined aspects found in these reviews.")
                with col2:
                    st.markdown("#### Rating vs. Text Discrepancy")
                    st.info("Hover over points to see review snippets.", icon="💡")
                    discrepancy_plot = px.scatter(reviews_with_scores, x="rating_display", y="text_polarity", color="discrepancy", color_continuous_scale=px.colors.sequential.Viridis, hover_data=['review_id', 'text'], labels={"rating_display": "Star Rating (Rounded)", "text_polarity": "Sentiment of Review Text", "discrepancy": "Discrepancy Score"}, title="Brighter points indicate higher discrepancy")
                    discrepancy_plot.update_xaxes(type='category', categoryorder='category ascending')
                    discrepancy_plot.update_layout(height=400)
                    st.plotly_chart(discrepancy_plot, use_container_width=True)
                
                st.markdown("---")
                st.markdown("#### Sentiment Over Time")
                streamgraph = alt.Chart(product_reviews).mark_area().encode(x=alt.X('yearmonth(timestamp):T', title='Month'), y=alt.Y('count():Q', stack='center', title='Review Volume'), color=alt.Color('sentiment:N', scale=alt.Scale(domain=['Positive', 'Neutral', 'Negative'], range=['#2ca02c', '#ff7f0e', '#d62728']), title='Sentiment'), tooltip=[alt.Tooltip('yearmonth(timestamp):T', title='Month'), alt.Tooltip('sentiment:N'), alt.Tooltip('count():Q', title='Reviews')]).properties(title="How sentiment has trended over the product's lifetime").interactive()
                st.altair_chart(streamgraph, use_container_width=True)

            with keyword_tab:
                st.subheader("Explore Keywords in Reviews")
                st.info("Use the filters to generate a word cloud and explore snippets from specific reviews.", icon="💡")
                # Keyword tab implementation...
                pass
        
    else:
        # --- MAIN SEARCH PAGE ---
        st.header("Search for Products")
        
        if 'search_results' not in st.session_state:
            st.session_state.search_results = pd.DataFrame()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            search_term = st.text_input("Search by product title:")
        with col2:
            available_categories = get_all_categories(conn)
            category = st.selectbox("Filter by Category", available_categories)
        with col3:
            sort_by = st.selectbox("Sort By", ["Popularity (Most Reviews)", "Highest Rating", "Lowest Rating"])
        
        if st.button("Search", type="primary"):
            st.session_state.page = 0 
            st.session_state.search_results = search_products(products_df, category, search_term, sort_by)
            st.rerun()

        if not st.session_state.search_results.empty:
            total_results = len(st.session_state.search_results)
            st.markdown("---")
            st.header(f"Search Results ({total_results} products found)")
            
            start_idx = st.session_state.page * PRODUCTS_PER_PAGE
            end_idx = start_idx + PRODUCTS_PER_PAGE
            paginated_results = st.session_state.search_results.iloc[start_idx:end_idx]

            for i in range(0, len(paginated_results), 4):
                cols = st.columns(4)
                for j in range(4):
                    if i + j < len(paginated_results):
                        row = paginated_results.iloc[i+j]
                        with cols[j].container(border=True):
                            # --- FIX: Use the image_url column directly ---
                            st.image(row['image_url'] or PLACEHOLDER_IMAGE_URL, use_container_width=True)
                            st.markdown(f"**{row['product_title']}**")
                            
                            avg_rating_text = f"{row['average_rating']:.2f}" if pd.notna(row['average_rating']) else "N/A"
                            review_count_text = int(row['review_count']) if pd.notna(row['review_count']) else 0

                            st.caption(f"Avg. Rating: {avg_rating_text} ⭐ ({review_count_text} reviews)")
                            
                            if st.button("View Details", key=row['parent_asin']):
                                st.session_state.selected_product = row['parent_asin']
                                st.rerun()
            st.markdown("---")
            
            # --- PAGINATION UI ---
            total_pages = (total_results + PRODUCTS_PER_PAGE - 1) // PRODUCTS_PER_PAGE
            if total_pages > 1:
                nav_cols = st.columns(3)
                if st.session_state.page > 0:
                    nav_cols[0].button("⬅️ Previous Page", on_click=lambda: st.session_state.update(page=st.session_state.page - 1))
                
                nav_cols[1].write(f"Page {st.session_state.page + 1} of {total_pages}")
                
                if (st.session_state.page + 1) < total_pages:
                    nav_cols[2].button("Next Page ➡️", on_click=lambda: st.session_state.update(page=st.session_state.page + 1))

else:
    st.error("Application setup failed. Please check logs and database connection.")

