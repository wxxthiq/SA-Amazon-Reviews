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
import logging
import kaggle

# --- Configure logging ---
logging.basicConfig(level=logging.INFO)

# --- Altair Data Transformer ---
alt.data_transformers.enable('default', max_rows=None)

# --- Configuration ---
KAGGLE_DATASET_SLUG = "wathiqsoualhi/mcauley-v3"
DATA_VERSION = 5 # Incremented to ensure a clean run
DATABASE_PATH = "amazon_reviews_images_v2.db" # This MUST match the unzipped file name
VERSION_FILE_PATH = ".db_version"
PRODUCTS_PER_PAGE = 16
PLACEHOLDER_IMAGE_URL = "https://via.placeholder.com/200"


# --- Data Loading Function ---
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

    if "KAGGLE_USERNAME" not in st.secrets or "KAGGLE_KEY" not in st.secrets:
        st.error("FATAL: Kaggle secrets not found. Please add KAGGLE_USERNAME and KAGGLE_KEY to your Streamlit secrets.")
        st.stop()
        
    kaggle_dir = os.path.expanduser("~/.kaggle")
    os.makedirs(kaggle_dir, exist_ok=True)
    kaggle_json_path = os.path.join(kaggle_dir, "kaggle.json")
    
    credentials = {
        "username": st.secrets["KAGGLE_USERNAME"],
        "key": st.secrets["KAGGLE_KEY"]
    }
    with open(kaggle_json_path, "w") as f:
        json.dump(credentials, f)
        
    os.chmod(kaggle_json_path, 0o600)

    try:
        with st.spinner(f"Downloading data for '{dataset_slug}' from Kaggle... Please wait."):
            logging.info(f"Attempting to download dataset: {dataset_slug}")
            kaggle.api.dataset_download_files(
                dataset=dataset_slug,
                path='.',
                unzip=True
            )
            logging.info("Kaggle API download successful.")

    except Exception as e:
        st.error(f"FATAL: An error occurred during the Kaggle API download: {e}")
        logging.error(f"Kaggle API error details: {e}")
        st.stop()
        
    if os.path.exists(db_path):
        with open(version_path, "w") as f:
            f.write(str(expected_version))
        st.success("Database download complete! Rerunning app...")
        st.rerun()
    else:
        st.error(f"FATAL: Download complete, but '{db_path}' was not found after unzipping. Please check the name of the file inside your Kaggle dataset's zip archive.")
        st.stop()


@st.cache_resource
def connect_to_db(path):
    try:
        return sqlite3.connect(path, uri=True, check_same_thread=False, timeout=15)
    except Exception as e:
        st.error(f"FATAL: Could not connect to database at '{path}'. Error: {e}")
        st.stop()

@st.cache_resource
def load_spacy_model():
    model_name = "en_core_web_sm"
    return spacy.load(model_name)

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
    
    # --- DETAILED PRODUCT VIEW (INTERACTIVE & MEMORY-EFFICIENT) ---
    if st.session_state.selected_product:
        selected_asin = st.session_state.selected_product
        
        # Load all data for the selected product ONCE
        @st.cache_data
        def load_product_data(_conn, asin):
            reviews_df = fetch_reviews_for_product(_conn, asin)
            if reviews_df.empty:
                return pd.DataFrame(), pd.DataFrame()
            reviews_with_scores = calculate_discrepancy(reviews_df)
            return reviews_df, reviews_with_scores

        raw_reviews, reviews_with_scores = load_product_data(conn, selected_asin)
        
        # Get static details from the main products summary
        product_details = products_df[products_df['parent_asin'] == selected_asin].iloc[0]

        # --- UI ---
        if st.button("⬅️ Back to Search Results"):
            st.session_state.selected_product = None
            st.session_state.image_index = 0
            st.rerun()

        st.header(product_details['product_title'])
        image_urls_str = product_details.get('image_urls')
        image_urls = image_urls_str.split(',') if pd.notna(image_urls_str) else []

        # Image Carousel
        if not image_urls:
            st.image(PLACEHOLDER_IMAGE_URL, use_container_width=True)
        else:
            # ... (Image carousel code as before)
            st.image(image_urls[st.session_state.image_index], use_container_width=True)

        if raw_reviews.empty:
            st.warning("No reviews found for this product.")
            st.stop()

        st.markdown("---")
        
        # --- ON-DEMAND HEAVY ANALYSIS ---
        st.subheader("Advanced Analysis")
        st.info("Heavy analysis is performed on-demand to ensure app stability.")
        
        if st.button("Run Full Analysis on Reviews"):
            with st.spinner("Running advanced analysis... This may take a moment on products with many reviews."):
                
                # Define aspects relevant to fashion
                fashion_aspects = ['fit', 'size', 'color', 'fabric', 'quality', 'price', 'style', 'comfort', 'stitching', 'zipper']
                absa_results_df = aspect_based_sentiment(reviews_with_scores, fashion_aspects, nlp)
                
                # --- Display results after computation ---
                col1, col2 = st.columns(2, gap="large")
                with col1:
                    st.markdown("#### Aspect Sentiment")
                    if not absa_results_df.empty:
                        absa_melted = absa_results_df.melt(id_vars=['aspect'], value_vars=['pos', 'neg'], var_name='sentiment', value_name='count')
                        absa_melted['count'] = absa_melted.apply(lambda row: -row['count'] if row['sentiment'] == 'neg' else row['count'], axis=1)
                        absa_chart = alt.Chart(absa_melted).mark_bar().encode(x=alt.X('count:Q', title='Number of Mentions'), y=alt.Y('aspect:N', sort='-x', title='Product Aspect'), color=alt.Color('sentiment:N')).properties(title="Positive vs. Negative Mentions by Feature")
                        st.altair_chart(absa_chart, use_container_width=True)
                    else:
                        st.warning("No defined aspects found in these reviews.")
                
                with col2:
                    st.markdown("#### Rating vs. Text Discrepancy")
                    discrepancy_plot = px.scatter(reviews_with_scores, x="rating_display", y="text_polarity", color="discrepancy", title="Brighter points indicate higher discrepancy")
                    st.plotly_chart(discrepancy_plot, use_container_width=True)

    # --- MAIN SEARCH PAGE ---
    else:
        st.header("Search for Products")
        
        col1, col2, col3 = st.columns(3)
        search_term = col1.text_input("Search by product title:")
        available_categories = get_all_categories(conn)
        category = col2.selectbox("Filter by Category", available_categories)
        sort_by = col3.selectbox("Sort By", ["Popularity (Most Reviews)", "Highest Rating", "Lowest Rating"])
        
        # The search logic should be applied directly to products_df
        search_results_df = products_df.copy()
        if search_term:
            search_results_df = search_results_df[search_results_df['product_title'].str.contains(search_term, case=False, na=False)]
        if category != "All":
            search_results_df = search_results_df[search_results_df['category'] == category]
        
        # Sorting logic
        if sort_by == "Popularity (Most Reviews)":
            search_results_df = search_results_df.sort_values(by="review_count", ascending=False)
        elif sort_by == "Highest Rating":
            search_results_df = search_results_df.sort_values(by="average_rating", ascending=False)
        elif sort_by == "Lowest Rating":
            search_results_df = search_results_df.sort_values(by="average_rating", ascending=True)

        st.markdown("---")
        total_results = len(search_results_df)
        st.header(f"Found {total_results} Products")
        
        start_idx = st.session_state.page * PRODUCTS_PER_PAGE
        end_idx = start_idx + PRODUCTS_PER_PAGE
        paginated_results = search_results_df.iloc[start_idx:end_idx]

        for i in range(0, len(paginated_results), 4):
            cols = st.columns(4)
            for j, col in enumerate(cols):
                if i + j < len(paginated_results):
                    row = paginated_results.iloc[i+j]
                    with col.container(border=True):
                        image_urls_str = row.get('image_urls')
                        thumbnail_url = image_urls_str.split(',')[0] if pd.notna(image_urls_str) else PLACEHOLDER_IMAGE_URL
                        st.image(thumbnail_url, use_container_width=True)
                        st.markdown(f"**{row['product_title']}**")
                        if st.button("View Details", key=row['parent_asin']):
                            st.session_state.selected_product = row['parent_asin']
                            st.rerun()

else:
    st.error("Application setup failed. Please check logs and database connection.")
