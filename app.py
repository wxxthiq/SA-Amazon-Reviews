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

# --- FINAL, CORRECTED CONFIGURATION ---
# This now points to the dataset and a NEW filename to bypass caching issues.
KAGGLE_DATASET_SLUG = "wathiqsoualhi/mcauley-lite" 
DATABASE_PATH = "amazon_reviews_lite_v4.db"  # <-- RENAMED FILE
DATA_VERSION = 3                             # <-- INCREMENTED VERSION

VERSION_FILE_PATH = ".db_version"
PRODUCTS_PER_PAGE = 16
REVIEWS_PER_PAGE = 5
PLACEHOLDER_IMAGE_URL = "https://via.placeholder.com/200"

# --- Data Loading Function (No changes needed here) ---
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
def connect_to_db(path, required_tables):
    """
    Connects to the SQLite database and verifies that all required tables exist.
    Includes a debug inspector to show what tables are actually present.
    """
    if not os.path.exists(path):
        st.error(f"Database file not found at path: {path}. Please ensure the download was successful.")
        st.stop()

    try:
        conn = sqlite3.connect(path, uri=True, check_same_thread=False, timeout=15)
        cursor = conn.cursor()
        
        # --- NEW DEBUG INSPECTOR ---
        with st.expander("🕵️‍♀️ Database Debug Inspector"):
            st.write(f"Checking database file: `{path}`")
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            existing_tables = {row[0] for row in cursor.fetchall()}
            
            if not existing_tables:
                st.warning("No tables found in the database file.")
            else:
                st.write("**Tables found:**")
                st.write(existing_tables)

            # Compare found tables with required tables
            missing_tables = set(required_tables) - existing_tables
            st.write("**Verification Result:**")
            if not missing_tables:
                st.success("✅ All required tables are present.")
            else:
                st.error(f"❌ Missing required tables: `{', '.join(missing_tables)}`")
        # --- END DEBUG INSPECTOR ---

        if missing_tables:
            st.error(
                f"Database Integrity Error: The application stopped because the database is missing required tables. "
                "Please check the debug inspector above for details."
            )
            st.stop()
            
        return conn
        
    except Exception as e:
        st.error(f"FATAL: Could not connect to or verify database at '{path}'. Error: {e}")
        st.stop()

# --- Data Fetching Functions (No changes needed here) ---

@st.cache_data
def get_product_summary_data(_conn):
    """Fetches the main product gallery data."""
    return pd.read_sql("SELECT * FROM products", _conn)

@st.cache_data
def get_discrepancy_data(_conn, asin):
    """Fetches the lightweight data for the discrepancy plot."""
    df = pd.read_sql("SELECT rating, text_polarity FROM discrepancy_data WHERE parent_asin = ?", _conn, params=(asin,))
    df['discrepancy'] = (df['text_polarity'] - ((df['rating'] - 3.0) / 2.0)).abs()
    return df

@st.cache_data
def get_rating_distribution_data(_conn, asin):
    """(NEW) Fetches the pre-computed rating distribution for a product."""
    return pd.read_sql("SELECT `1_star`, `2_star`, `3_star`, `4_star`, `5_star` FROM rating_distribution WHERE parent_asin = ?", _conn, params=(asin,))

def get_paginated_reviews(_conn, asin, page_num, page_size):
    """Fetches a small 'page' of raw reviews to display."""
    offset = (page_num - 1) * page_size
    return pd.read_sql(f"SELECT rating, sentiment, text FROM reviews WHERE parent_asin = ? LIMIT ? OFFSET ?", _conn, params=(asin, page_size, offset))


# --- Main App ---
st.set_page_config(layout="wide", page_title="Amazon Review Explorer")
st.title("⚡ Amazon Reviews - Sentiment Dashboard (Lite Version)")

REQUIRED_TABLES = ['products', 'reviews', 'discrepancy_data', 'rating_distribution']

download_data_with_versioning(KAGGLE_DATASET_SLUG, DATABASE_PATH, VERSION_FILE_PATH, DATA_VERSION)
conn = connect_to_db(DATABASE_PATH, REQUIRED_TABLES)


# Initialize session state
if 'page' not in st.session_state: st.session_state.page = 0
if 'review_page' not in st.session_state: st.session_state.review_page = 1
if 'selected_product' not in st.session_state: st.session_state.selected_product = None

if conn:
    products_df = get_product_summary_data(conn)
    
    # --- DETAILED PRODUCT VIEW ---
    if st.session_state.selected_product:
        selected_asin = st.session_state.selected_product
        product_details = products_df.loc[products_df['parent_asin'] == selected_asin].iloc[0]

        if st.button("⬅️ Back to Search"):
            st.session_state.selected_product = None
            st.session_state.review_page = 1
            st.rerun()

        st.header(product_details['product_title'])
        # ... (Image carousel code can remain here)

        st.markdown("---")
        
        # --- Visualization Tabs ---
        vis_tab, reviews_tab = st.tabs(["📊 Sentiment Analysis", "💬 Individual Reviews"])

        with vis_tab:
            st.subheader("Sentiment Analysis Visualizations")
            
            discrepancy_df = get_discrepancy_data(conn, selected_asin)
            rating_dist_df = get_rating_distribution_data(conn, selected_asin)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Rating Distribution")
                if not rating_dist_df.empty:
                    # Reshape the data for Altair
                    dist_data = rating_dist_df.T.reset_index()
                    dist_data.columns = ['Star Rating', 'Count']
                    dist_data['Star Rating'] = dist_data['Star Rating'].str.replace('_', ' ')
                    
                    chart = alt.Chart(dist_data).mark_bar().encode(
                        x=alt.X('Star Rating', sort=None),
                        y=alt.Y('Count'),
                        tooltip=['Star Rating', 'Count']
                    ).properties(title="Overall Rating Distribution")
                    st.altair_chart(chart, use_container_width=True)
                else:
                    st.warning("No rating distribution data available.")
            
            with col2:
                st.markdown("#### Rating vs. Text Discrepancy")
                if not discrepancy_df.empty:
                    plot = px.scatter(discrepancy_df, x="rating", y="text_polarity", color="discrepancy", title="Discrepancy Plot")
                    st.plotly_chart(plot, use_container_width=True)
                else:
                    st.warning("No discrepancy data available.")

        with reviews_tab:
            st.subheader("Paginated Individual Reviews")
            reviews_df = get_paginated_reviews(conn, selected_asin, st.session_state.review_page, REVIEWS_PER_PAGE)
            if not reviews_df.empty:
                for index, row in reviews_df.iterrows():
                    st.markdown(f"**Rating: {row['rating']} ⭐ | Sentiment: {row['sentiment']}**")
                    st.markdown(f"> {row['text']}")
                    st.divider()
                
                col1, col2, col3 = st.columns([1, 8, 1])
                if st.session_state.review_page > 1:
                    if col1.button("⬅️ Previous"):
                        st.session_state.review_page -= 1
                        st.rerun()
                if len(reviews_df) == REVIEWS_PER_PAGE:
                     if col3.button("Next ➡️"):
                        st.session_state.review_page += 1
                        st.rerun()
            else:
                st.warning("No more reviews to display.")

    # --- MAIN SEARCH PAGE ---
    else:
        st.session_state.review_page = 1
        st.header("Search for Products")
        # ... (Your existing main page search and gallery code is fine here) ...

else:
    st.error("Application setup failed. Please check database connection.")
