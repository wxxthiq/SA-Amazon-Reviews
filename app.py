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
import json
import plotly.express as px

# --- Altair Data Transformer ---
alt.data_transformers.enable('default', max_rows=None)

# --- Configuration ---
# IMPORTANT: The slug should NOT contain your username for this method to work best
KAGGLE_DATASET_SLUG = "wathiqsoualhi/mcauley-v3" 
DATA_VERSION = 3 # Increment version to ensure new download attempt
DATABASE_PATH = "amazon_reviews_images_v2.db"
VERSION_FILE_PATH = ".db_version"
PRODUCTS_PER_PAGE = 16
PLACEHOLDER_IMAGE_URL = "https://via.placeholder.com/200"

# --- NEW, ROBUST Data Loading Function using the official Kaggle library ---
def download_data_with_versioning(dataset_slug, db_path, version_path, expected_version):
    """Downloads data using the official Kaggle API library and handles authentication."""
    current_version = -1
    if os.path.exists(version_path):
        with open(version_path, "r") as f:
            try:
                current_version = int(f.read().strip())
            except (ValueError, TypeError):
                current_version = -1
    
    if current_version == expected_version and os.path.exists(db_path):
        logging.info("Database is up to date.")
        return

    st.info(f"Database v{current_version} is outdated (expected v{expected_version}). Forcing fresh download...")
    if os.path.exists(db_path): os.remove(db_path)
    if os.path.exists(version_path): os.remove(version_path)

    # 1. Check for secrets
    if "KAGGLE_USERNAME" not in st.secrets or "KAGGLE_KEY" not in st.secrets:
        st.error("FATAL: Kaggle secrets not found. Please add KAGGLE_USERNAME and KAGGLE_KEY to your Streamlit secrets.")
        st.stop()
        
    # 2. Create the .kaggle directory and kaggle.json file at runtime
    kaggle_dir = os.path.expanduser("~/.kaggle")
    os.makedirs(kaggle_dir, exist_ok=True)
    kaggle_json_path = os.path.join(kaggle_dir, "kaggle.json")
    
    credentials = {
        "username": st.secrets["KAGGLE_USERNAME"],
        "key": st.secrets["KAGGLE_KEY"]
    }
    with open(kaggle_json_path, "w") as f:
        json.dump(credentials, f)
        
    # 3. Set correct file permissions (important for the Kaggle library)
    os.chmod(kaggle_json_path, 0o600)

    # 4. Use the Kaggle API to download the dataset
    with st.spinner(f"Downloading data for '{dataset_slug}' from Kaggle... Please wait."):
        try:
            logging.info(f"Attempting to download dataset: {dataset_slug}")
            # The official Kaggle API for downloading datasets
            kaggle.api.dataset_download_files(
                dataset=dataset_slug,
                path='.',  # Download to the current directory
                unzip=True
            )
            logging.info("Kaggle API download successful.")
            
            # 5. Verify download and update version file
            if os.path.exists(db_path):
                with open(version_path, "w") as f:
                    f.write(str(expected_version))
                st.success("Database download complete! Rerunning app...")
                st.rerun()
            else:
                st.error(f"FATAL: Download complete, but '{db_path}' was not found after unzipping. Please check the name of the file inside your Kaggle dataset's zip archive.")
                st.stop()

        except Exception as e:
            # Catch potential errors from the Kaggle library, including authentication
            st.error(f"FATAL: An error occurred during the Kaggle API download: {e}")
            logging.error(f"Kaggle API error details: {e}")
            st.stop()


@st.cache_resource
def connect_to_db(path):
    try:
        return sqlite3.connect(path, uri=True, check_same_thread=False, timeout=15)
    except Exception as e:
        st.error(f"FATAL: Could not connect to database at '{path}'. Error: {e}")
        st.stop()

# --- THIS IS THE CORRECTED FUNCTION ---
@st.cache_resource
def load_spacy_model():
    """
    Loads the spaCy model, which is guaranteed to be installed via requirements.txt.
    """
    model_name = "en_core_web_sm"
    return spacy.load(model_name)

# --- Data Fetching Functions ---
@st.cache_data
def get_all_categories(_conn):
    if _conn is None: return ["All"]
    try:
        df = pd.read_sql_query("SELECT DISTINCT category FROM products", _conn)
        categories = sorted(df['category'].dropna().unique().tolist())
        categories.insert(0, "All")
        return categories
    except Exception as e:
        st.warning(f"Could not load categories: {e}")
        return ["All"]

@st.cache_data
def get_product_summary_data(_conn):
    if _conn is None: return pd.DataFrame()
    try:
        query = "SELECT * FROM products"
        df = pd.read_sql_query(query, _conn)
        return df
    except Exception as e:
        st.error(f"An error occurred while loading product summary: {e}")
        return pd.DataFrame()

def fetch_reviews_for_product(_conn, asin):
    if _conn is None: return pd.DataFrame()
    query = "SELECT * FROM reviews WHERE parent_asin = ?"
    df = pd.read_sql_query(query, _conn, params=(asin,))
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.reset_index().rename(columns={'index': 'review_id'})
    return df

# --- Analysis Functions ---
@st.cache_data
def calculate_discrepancy(reviews_df):
    if 'text_polarity' not in reviews_df.columns:
        st.error("Data integrity error: text_polarity column not found in reviews.")
        return reviews_df.assign(discrepancy=0, rating_display=reviews_df['rating'])
    df = reviews_df.copy()
    df['normalized_rating'] = (df['rating'] - 3) / 2
    df['discrepancy'] = (df['text_polarity'] - df['normalized_rating']).abs()
    df['rating_display'] = df['rating'].round().astype(int)
    return df

@st.cache_data
def aspect_based_sentiment(_reviews_df, aspects, _nlp_model):
    aspect_sentiments = {aspect: {'pos': 0, 'neg': 0, 'neu': 0, 'mentions': 0} for aspect in aspects}
    for review_text in _reviews_df['text'].dropna():
        try:
            for sentence in _nlp_model(review_text).sents:
                for aspect in aspects:
                    if f' {aspect.lower()} ' in f' {sentence.text.lower()} ':
                        aspect_sentiments[aspect]['mentions'] += 1
                        sentiment = TextBlob(sentence.text).sentiment.polarity
                        if sentiment > 0.1: aspect_sentiments[aspect]['pos'] += 1
                        elif sentiment < -0.1: aspect_sentiments[aspect]['neg'] += 1
                        else: aspect_sentiments[aspect]['neu'] += 1
        except Exception:
            continue
    results = [{'aspect': aspect, **scores} for aspect, scores in aspect_sentiments.items() if scores['mentions'] > 0]
    return pd.DataFrame(results)

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

# The KAGGLE_USERNAME is no longer needed here, it's read from secrets
download_data_with_versioning(KAGGLE_DATASET_SLUG, DATABASE_PATH, VERSION_FILE_PATH, DATA_VERSION)
conn = connect_to_db(DATABASE_PATH)
nlp = load_spacy_model()

# Initialize session state variables
if 'page' not in st.session_state: st.session_state.page = 0
if 'selected_product' not in st.session_state: st.session_state.selected_product = None
if 'image_index' not in st.session_state: st.session_state.image_index = 0

# --- Main App Logic ---
if conn and nlp:
    products_df = get_product_summary_data(conn)
    
    # --- DETAILED PRODUCT VIEW (INTERACTIVE) ---
    if st.session_state.selected_product:
        selected_asin = st.session_state.selected_product
        
        @st.cache_data
        def load_product_data(_conn, asin):
            reviews_df = fetch_reviews_for_product(_conn, asin)
            reviews_with_scores = calculate_discrepancy(reviews_df)
            return reviews_with_scores

        product_reviews_df = load_product_data(conn, selected_asin)
        product_details = products_df[products_df['parent_asin'] == selected_asin].iloc[0]

        if st.button("⬅️ Back to Search Results"):
            st.session_state.selected_product = None
            st.session_state.image_index = 0
            st.rerun()
        
        st.header(product_details['product_title'])
        image_urls_str = product_details.get('image_urls')
        image_urls = image_urls_str.split(',') if pd.notna(image_urls_str) else []

        if not image_urls:
            st.image(PLACEHOLDER_IMAGE_URL, use_container_width=True)
        else:
            if st.session_state.image_index >= len(image_urls): st.session_state.image_index = 0
            def next_image(): st.session_state.image_index = (st.session_state.image_index + 1) % len(image_urls)
            def prev_image(): st.session_state.image_index = (st.session_state.image_index - 1 + len(image_urls)) % len(image_urls)

            st.image(image_urls[st.session_state.image_index], use_container_width=True)
            if len(image_urls) > 1:
                col1, col2, col3 = st.columns([1, 8, 1])
                with col1: st.button("⬅️", on_click=prev_image, use_container_width=True, key="prev_img")
                with col2: st.caption(f"Image {st.session_state.image_index + 1} of {len(image_urls)}")
                with col3: st.button("➡️", on_click=next_image, use_container_width=True, key="next_img")
        
        st.markdown("---")
        st.header("Interactive Review Analysis")
        
        filter_cols = st.columns([1, 1, 2])
        selected_ratings = filter_cols[0].multiselect('Filter by Star Rating', options=sorted(product_reviews_df['rating'].unique()), default=sorted(product_reviews_df['rating'].unique()))
        min_date = product_reviews_df['timestamp'].min().date() if not product_reviews_df.empty else None
        max_date = product_reviews_df['timestamp'].max().date() if not product_reviews_df.empty else None
        
        selected_date_range = (min_date, max_date)
        if min_date and max_date: selected_date_range = filter_cols[1].date_input("Filter by Review Date", value=(min_date, max_date))

        keyword_search = filter_cols[2].text_input("Search for a keyword in review text")

        filtered_df = product_reviews_df
        if selected_ratings: filtered_df = filtered_df[filtered_df['rating'].isin(selected_ratings)]
        if len(selected_date_range) == 2: filtered_df = filtered_df[(filtered_df['timestamp'].dt.date >= selected_date_range[0]) & (filtered_df['timestamp'].dt.date <= selected_date_range[1])]
        if keyword_search: filtered_df = filtered_df[filtered_df['text'].str.contains(keyword_search, case=False, na=False)]

        st.metric("Filtered Reviews", f"{len(filtered_df)} of {len(product_reviews_df)}")
        st.markdown("---")

        if filtered_df.empty:
            st.warning("No reviews match your current filter criteria.")
        else:
            overview_tab, keyword_tab = st.tabs(["📊 Comprehensive Analysis", "🔎 Keyword Explorer"])
            with overview_tab:
                st.plotly_chart(px.scatter(filtered_df, x="rating_display", y="text_polarity", color="discrepancy", title="Rating vs. Text Sentiment Discrepancy"), use_container_width=True)
                if st.button("Run Aspect Analysis on Filtered Reviews"):
                    fashion_aspects = ['fit', 'size', 'color', 'fabric', 'quality', 'price', 'style', 'comfort']
                    absa_results_df = aspect_based_sentiment(filtered_df, fashion_aspects, nlp)
                    if not absa_results_df.empty:
                        absa_melted = absa_results_df.melt(id_vars=['aspect'], value_vars=['pos', 'neg'], var_name='sentiment', value_name='count')
                        absa_melted['count'] = absa_melted.apply(lambda row: -row['count'] if row['sentiment'] == 'neg' else row['count'], axis=1)
                        absa_chart = alt.Chart(absa_melted).mark_bar().encode(x=alt.X('count:Q', title='Number of Mentions'), y=alt.Y('aspect:N', sort='-x'), color=alt.Color('sentiment:N')).properties(title="Aspect Sentiment")
                        st.altair_chart(absa_chart, use_container_width=True)
                    else:
                        st.info("No defined aspects found in filtered reviews.")
            with keyword_tab:
                full_text = " ".join(filtered_df['text'].dropna().astype(str))
                st.pyplot(generate_wordcloud(full_text, "Top Words in Filtered Reviews", []))

    # --- MAIN SEARCH PAGE ---
    else:
        st.header("Search for Products")
        search_df = get_product_summary_data(conn) 

        col1, col2, col3 = st.columns(3)
        search_term = col1.text_input("Search by product title:")
        available_categories = get_all_categories(conn)
        category = col2.selectbox("Filter by Category", available_categories)
        sort_by = col3.selectbox("Sort By", ["Popularity (Most Reviews)", "Highest Rating", "Lowest Rating"])
        
        if search_term: search_df = search_df[search_df['product_title'].str.contains(search_term, case=False, na=False)]
        if category != "All": search_df = search_df[search_df['category'] == category]
        
        if sort_by == "Popularity (Most Reviews)": search_df = search_df.sort_values(by="review_count", ascending=False)
        elif sort_by == "Highest Rating": search_df = search_df.sort_values(by="average_rating", ascending=False)
        elif sort_by == "Lowest Rating": search_df = search_df.sort_values(by="average_rating", ascending=True)

        st.markdown("---")
        total_results = len(search_df)
        st.header(f"Found {total_results} Products")
        
        start_idx = st.session_state.page * PRODUCTS_PER_PAGE
        end_idx = start_idx + PRODUCTS_PER_PAGE
        paginated_results = search_df.iloc[start_idx:end_idx]

        for i in range(0, len(paginated_results), 4):
            cols = st.columns(4)
            for j, col in enumerate(cols):
                if i + j < len(paginated_results):
                    row = paginated_results.iloc[i+j]
                    with col.container(border=True):
                        image_urls_str = row.get('image_urls')
                        if pd.notna(image_urls_str): thumbnail_url = image_urls_str.split(',')[0]
                        else: thumbnail_url = PLACEHOLDER_IMAGE_URL
                        st.image(thumbnail_url, use_container_width=True)
                        st.markdown(f"**{row['product_title']}**")
                        if st.button("View Details", key=row['parent_asin']):
                            st.session_state.selected_product = row['parent_asin']
                            st.rerun()

else:
    st.error("Application setup failed. Please check logs and database connection.")
