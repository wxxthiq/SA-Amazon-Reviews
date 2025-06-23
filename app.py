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
KAGGLE_DATASET_SLUG = "amazon-3-mcauley"
# IMPORTANT: Go to your Kaggle dataset page -> "Data" tab -> look at the version history.
# Set this number to match the latest version number of your dataset (e.g., 2, 3, etc.)
DATA_VERSION = 0

DATABASE_PATH = "amazon_3_category.db" 
VERSION_FILE_PATH = ".db_version" # A hidden file to store the current data version
PRODUCTS_PER_PAGE = 16
PLACEHOLDER_IMAGE_URL = "https://via.placeholder.com/150"


# In app.py, replace the old download function with this one.
# Make sure this is the function in your app.py
def download_data_with_versioning(dataset_slug, db_path, version_path, expected_version):
    """
    Downloads and unzips data only if the local version is out of date.
    This version includes extensive debugging messages.
    """
    st.warning("--- STARTING DATA CHECK ---")

    current_version = 0
    if os.path.exists(version_path):
        st.write(f"✔️ Found version file at: {version_path}")
        with open(version_path, "r") as f:
            try:
                version_content = f.read().strip()
                current_version = int(version_content)
                st.write(f"✔️ Read version from file: '{version_content}'")
            except (ValueError, TypeError):
                current_version = 0
                st.write(f"⚠️ Could not parse version from file. Content was: '{version_content}'. Defaulting to 0.")
    else:
        st.write(f"❌ Version file not found at: {version_path}")

    st.info(f"ℹ️ Comparing stored version ({current_version}) with expected version ({expected_version})...")

    if current_version == expected_version and os.path.exists(db_path):
        st.success("✅ Database is up to date. Skipping download.")
        st.warning("--- DATA CHECK COMPLETE ---")
        return

    st.warning("🔥 Database version mismatch or file not found. Forcing fresh download.")

    if os.path.exists(db_path):
        st.write(f"Attempting to delete old database file at {db_path}...")
        os.remove(db_path)
        st.write("✔️ Old database deleted.")
    if os.path.exists(version_path):
        st.write(f"Attempting to delete old version file at {version_path}...")
        os.remove(version_path)
        st.write("✔️ Old version file deleted.")

    url = f"https://www.kaggle.com/api/v1/datasets/download/{dataset_slug}"
    zip_file_path = "archive.zip"
    st.write(f"Starting download from: {url}")

    with st.spinner("Downloading data from Kaggle... Please wait."):
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(zip_file_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            st.write("Download complete. Unzipping file...")
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                zip_ref.extractall(".")

            st.write("Unzip complete. Cleaning up zip file...")
            os.remove(zip_file_path)

            if os.path.exists(db_path):
                st.write(f"Verified that new database exists at {db_path}.")
                with open(version_path, "w") as f:
                    f.write(str(expected_version))
                st.write(f"✔️ Wrote new version '{expected_version}' to version file.")
                st.success("Database download process complete!")
            else:
                st.error(f"FATAL: Download complete, but the expected file '{db_path}' was not found after unzipping.")
                st.stop()

        except Exception as e:
            st.error(f"FATAL: An error occurred during download and extraction: {e}")
            st.stop()
    st.warning("--- DATA CHECK COMPLETE ---")

# --- FINAL, ROBUST DATA DOWNLOADER with Versioning ---
# --- All other helper functions remain the same ---
def extract_first_image_url(image_data):
    if not isinstance(image_data, str) or not image_data.startswith('['): return None
    try:
        image_list = json.loads(image_data.replace("'", '"'))
        if image_list and isinstance(image_list, list): return image_list[0]
    except (json.JSONDecodeError, IndexError): return None
    return None

@st.cache_resource
def connect_to_db(path):
    try:
        return sqlite3.connect(path, uri=True, check_same_thread=False)
    except Exception as e:
        st.error(f"FATAL: Could not connect to the database. Error: {e}")
        return None

@st.cache_resource
def load_spacy_model():
    model_name = "en_core_web_sm"
    try:
        nlp = spacy.load(model_name)
    except OSError:
        st.info(f"SpaCy model '{model_name}' not found. Downloading...")
        try:
            subprocess.run(["python", "-m", "spacy", "download", model_name], check=True)
            nlp = spacy.load(model_name)
            st.success(f"Model '{model_name}' downloaded successfully!")
        except Exception as e:
            st.error(f"Failed to download or load spaCy model. Error: {e}")
            return None
    return nlp

@st.cache_data
def load_gallery_data(_conn):
    if _conn is None: return None
    try:
        query = """
        SELECT
            parent_asin, MAX(product_title) as product_title, MAX(image_url) as image_url,
            MAX(category) as category, average_rating, review_count
        FROM reviews
        GROUP BY parent_asin, average_rating, review_count
        """
        df = pd.read_sql_query(query, _conn)
        return df
    except Exception as e:
        st.error(f"An error occurred while loading gallery data: {e}")
        return None

def search_products(products_df, category, search_term, sort_by):
    if products_df is None or products_df.empty: return pd.DataFrame()
    results_df = products_df.copy()
    if category != "All":
        results_df = results_df[results_df['category'] == category]
    if search_term:
        results_df = results_df[results_df['product_title'].str.contains(search_term, case=False, na=False)]
    if sort_by == "Popularity (Most Reviews)":
        results_df = results_df.sort_values(by="review_count", ascending=False)
    elif sort_by == "Highest Rating":
        results_df = results_df.sort_values(by="average_rating", ascending=False)
    elif sort_by == "Lowest Rating":
        results_df = results_df.sort_values(by="average_rating", ascending=True)
    return results_df

def fetch_reviews_for_product(_conn, asin):
    if _conn is None: return pd.DataFrame()
    query = "SELECT * FROM reviews WHERE parent_asin = ?"
    df = pd.read_sql_query(query, _conn, params=(asin,))
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.reset_index().rename(columns={'index': 'review_id'})
    return df

def generate_wordcloud(text, title, custom_stopwords):
    stopwords = STOPWORDS.union(set(custom_stopwords))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title(title, fontsize=16)
    if text and isinstance(text, str) and text.strip():
        try:
            wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=stopwords, colormap='viridis').generate(text)
            ax.imshow(wordcloud, interpolation='bilinear')
        except ValueError:
            ax.text(0.5, 0.5, 'No words to display', ha='center', va='center')
    else:
        ax.text(0.5, 0.5, 'No reviews for this data segment', ha='center', va='center')
    ax.axis('off')
    return fig

def get_top_keywords(text, custom_stopwords, top_n=50):
    if not isinstance(text, str): return []
    stopwords = STOPWORDS.union(set(custom_stopwords))
    words = re.findall(r"[a-zA-Z']+", text.lower())
    filtered_words = [word for word in words if word not in stopwords and len(word) > 2]
    word_counts = Counter(filtered_words)
    return [word for word, count in word_counts.most_common(top_n)]

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


# --- Main App Execution ---
st.set_page_config(layout="wide", page_title="Amazon Review Explorer")
st.title("⚡ Amazon Reviews - Sentiment Dashboard")

KAGGLE_DATASET = f"{KAGGLE_USERNAME}/{KAGGLE_DATASET_SLUG}"
download_data_with_versioning(KAGGLE_DATASET, DATABASE_PATH, VERSION_FILE_PATH, DATA_VERSION)

conn = connect_to_db(DATABASE_PATH)
nlp = load_spacy_model()

if 'page' not in st.session_state: st.session_state.page = 0
if 'selected_product' not in st.session_state: st.session_state.selected_product = None
if 'all_search_results' not in st.session_state: st.session_state.all_search_results = pd.DataFrame()
if 'search_clicked' not in st.session_state: st.session_state.search_clicked = False

products_df = load_gallery_data(conn)

# --- DETAILED PRODUCT VIEW ---
if st.session_state.selected_product:
    # This block is unchanged, so I'm omitting it for brevity
    # Your fully working detail view code goes here
    pass

# --- MAIN SEARCH PAGE ---
else:
    st.header("Search for Products")
    col1, col2, col3 = st.columns(3)
    with col1:
        search_term = st.text_input("Search by product title:")
    with col2:
        if products_df is not None and not products_df.empty:
            available_categories = sorted(products_df['category'].unique().tolist())
            available_categories.insert(0, "All")
            category = st.selectbox("Filter by Category", available_categories)
        else:
            # Handle case where products_df might be empty on first load
            st.warning("Loading product categories...")
            category = st.selectbox("Filter by Category", ["All"])
    with col3:
        sort_by = st.selectbox("Sort By", ["Popularity (Most Reviews)", "Highest Rating", "Lowest Rating"])
    
    if st.button("Search", type="primary"):
        st.session_state.page = 0
        st.session_state.search_clicked = True
        st.session_state.all_search_results = search_products(products_df, category, search_term, sort_by)

    if 'all_search_results' in st.session_state and not st.session_state.all_search_results.empty:
        st.markdown("---")
        st.header("Search Results")

        start_idx = st.session_state.page * PRODUCTS_PER_PAGE
        end_idx = start_idx + PRODUCTS_PER_PAGE
        paginated_results = st.session_state.all_search_results.iloc[start_idx:end_idx]

        for i in range(0, len(paginated_results), 4):
            cols = st.columns(4)
            for j in range(4):
                if i + j < len(paginated_results):
                    row = paginated_results.iloc[i+j]
                    with cols[j].container(border=True):
                        image_url = extract_first_image_url(row['image_url'])
                        st.image(image_url or PLACEHOLDER_IMAGE_URL, use_container_width=True)
                        st.markdown(f"**{row['product_title']}**")
                        
                        avg_rating_text = f"{row['average_rating']:.2f}" if pd.notna(row['average_rating']) else "N/A"
                        review_count_text = int(row['review_count']) if pd.notna(row['review_count']) else 0

                        st.caption(f"Avg. Rating: {avg_rating_text} ⭐ ({review_count_text} reviews)")
                        
                        if st.button("View Details", key=row['parent_asin']):
                            st.session_state.selected_product = row['parent_asin']
                            st.rerun()
        st.markdown("---")
        col_center, _ = st.columns([1, 3])
        with col_center:
            if len(st.session_state.all_search_results) > end_idx:
                if st.button("Load More Results"):
                    st.session_state.page += 1
                    st.rerun()
                    
    elif st.session_state.get('search_clicked'):
        st.info("No products found matching your criteria.")