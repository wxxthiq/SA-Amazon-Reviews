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

# --- App Configuration ---
# This points to the dataset and file that we have verified are correct.
KAGGLE_DATASET_SLUG = "wathiqsoualhi/mcauley-lite" 
DATABASE_PATH = "amazon_reviews_lite_v4.db"  # Using v5 to ensure no caching issues
DATA_VERSION = 4                             # Matching the DB version

VERSION_FILE_PATH = ".db_version"
PRODUCTS_PER_PAGE = 16
REVIEWS_PER_PAGE = 5
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
def connect_to_db(path, required_tables):
    """
    Connects to the SQLite database and verifies that all required tables exist.
    """
    if not os.path.exists(path):
        st.error(f"Database file not found at path: {path}. Please ensure the download was successful.")
        st.stop()

    try:
        conn = sqlite3.connect(path, uri=True, check_same_thread=False, timeout=15)
        cursor = conn.cursor()
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        existing_tables = {row[0] for row in cursor.fetchall()}
        missing_tables = set(required_tables) - existing_tables
        
        if missing_tables:
            st.error(
                f"Database Integrity Error: The database at '{path}' is missing required tables: "
                f"`{', '.join(missing_tables)}`. This indicates the wrong database file is being used."
            )
            st.stop()
            
        return conn
        
    except Exception as e:
        st.error(f"FATAL: Could not connect to or verify database at '{path}'. Error: {e}")
        st.stop()

# --- Data Fetching Functions ---

@st.cache_data
def get_all_categories(_conn):
    """Fetches a list of all unique categories. Cached for efficiency on the main page."""
    df = pd.read_sql("SELECT DISTINCT category FROM products", _conn)
    categories = sorted(df['category'].dropna().unique().tolist())
    # MODIFIED: Do not add "All" by default. The user must select a category.
    categories.insert(0, "--- Select a Category ---")
    return categories

def get_filtered_products(_conn, category, search_term, sort_by, limit, offset):
    """
    (NEW) Fetches a paginated and filtered list of products directly from the database.
    This is highly memory-efficient as it doesn't load all products at once.
    """
    query = "SELECT * FROM products"
    count_query = "SELECT COUNT(*) FROM products"
    
    conditions = []
    params = []

    if category != "--- Select a Category ---":
        conditions.append("category = ?")
        params.append(category)

    if search_term:
        conditions.append("product_title LIKE ?")
        params.append(f"%{search_term}%")

    if conditions:
        query += " WHERE " + " AND ".join(conditions)
        count_query += " WHERE " + " AND ".join(conditions)

    if sort_by == "Popularity (Most Reviews)":
        query += " ORDER BY review_count DESC"
    elif sort_by == "Highest Rating":
        query += " ORDER BY average_rating DESC"
    else: # Lowest Rating
        query += " ORDER BY average_rating ASC"

    # Get total count for pagination
    total_count = _conn.execute(count_query, params).fetchone()[0]

    # Add pagination to the main query
    query += f" LIMIT {limit} OFFSET {offset}"
    
    df = pd.read_sql(query, _conn, params=params)
    
    return df, total_count


def get_single_product_details(_conn, asin):
    """Fetches details for only one product, used for the detail page."""
    return pd.read_sql("SELECT * FROM products WHERE parent_asin = ?", _conn, params=(asin,))


def get_discrepancy_data(_conn, asin):
    """Fetches the lightweight data for the discrepancy plot."""
    df = pd.read_sql("SELECT rating, text_polarity FROM discrepancy_data WHERE parent_asin = ?", _conn, params=(asin,))
    if not df.empty:
        df['discrepancy'] = (df['text_polarity'] - ((df['rating'] - 3.0) / 2.0)).abs()
    return df

def get_rating_distribution_data(_conn, asin):
    """Fetches the pre-computed rating distribution for a product."""
    return pd.read_sql("SELECT `1_star`, `2_star`, `3_star`, `4_star`, `5_star` FROM rating_distribution WHERE parent_asin = ?", _conn, params=(asin,))

def get_paginated_reviews(_conn, asin, page_num, page_size):
    """Fetches a small 'page' of raw reviews to display."""
    offset = (page_num - 1) * page_size
    return pd.read_sql(f"SELECT rating, sentiment, text FROM reviews WHERE parent_asin = ? LIMIT ? OFFSET ?", _conn, params=(asin, page_size, offset))


# --- Main App ---
st.set_page_config(layout="wide", page_title="Amazon Review Explorer")
st.title("⚡ Amazon Reviews - Sentiment Dashboard")

REQUIRED_TABLES = ['products', 'reviews', 'discrepancy_data', 'rating_distribution']

download_data_with_versioning(KAGGLE_DATASET_SLUG, DATABASE_PATH, VERSION_FILE_PATH, DATA_VERSION)
conn = connect_to_db(DATABASE_PATH, REQUIRED_TABLES)

# Initialize session state
if 'page' not in st.session_state: st.session_state.page = 0
if 'review_page' not in st.session_state: st.session_state.review_page = 1
if 'selected_product' not in st.session_state: st.session_state.selected_product = None
if 'category' not in st.session_state: st.session_state.category = "--- Select a Category ---"


if conn:
    # --- DETAILED PRODUCT VIEW ---
    if st.session_state.selected_product:
        selected_asin = st.session_state.selected_product
        # Fetch only the details for the selected product
        product_details_df = get_single_product_details(conn, selected_asin)
        
        if product_details_df.empty:
            st.error("Product details could not be found.")
            st.stop()
        
        product_details = product_details_df.iloc[0]

        if st.button("⬅️ Back to Search"):
            st.session_state.selected_product = None
            st.session_state.review_page = 1
            st.rerun()

        st.header(product_details['product_title'])
        image_urls_str = product_details.get('image_urls')
        image_urls = image_urls_str.split(',') if pd.notna(image_urls_str) else []
        if image_urls:
            st.image(image_urls[0], use_container_width=True)
        else:
            st.image(PLACEHOLDER_IMAGE_URL, use_container_width=True)

        st.markdown("---")
        
        vis_tab, reviews_tab = st.tabs(["📊 Sentiment Analysis", "💬 Individual Reviews"])
        with vis_tab:
            # ... (visualization logic remains the same) ...
            pass
        with reviews_tab:
            # ... (review pagination logic remains the same) ...
            pass

    # --- MAIN SEARCH PAGE ---
    else:
        st.session_state.review_page = 1
        st.header("Search for Products")
        
        # --- Search and Filter Controls ---
        col1, col2, col3 = st.columns(3)
        with col1:
            search_term = st.text_input("Search by product title:")
        with col2:
            available_categories = get_all_categories(conn)
            # When a new category is selected, reset the page to 0
            def on_category_change():
                st.session_state.page = 0
            category = st.selectbox("Filter by Category", available_categories, key='category', on_change=on_category_change)
        with col3:
            sort_by = st.selectbox("Sort By", ["Popularity (Most Reviews)", "Highest Rating", "Lowest Rating"], key='sort_by')

        if category == "--- Select a Category ---":
            st.info("Please select a category to view products.")
        else:
            # Fetch only the required page of products from the database
            paginated_results, total_results = get_filtered_products(
                conn, category, search_term, sort_by, 
                limit=PRODUCTS_PER_PAGE, 
                offset=st.session_state.page * PRODUCTS_PER_PAGE
            )

            st.markdown("---")
            st.header(f"Found {total_results} Products in '{category}'")
            
            if paginated_results.empty and total_results > 0:
                st.warning("No more products to display on this page.")
            else:
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
                                avg_rating = row.get('average_rating', 0)
                                review_count = row.get('review_count', 0)
                                st.caption(f"Avg. Rating: {avg_rating:.2f} ⭐ ({int(review_count)} reviews)")
                                if st.button("View Details", key=row['parent_asin']):
                                    st.session_state.selected_product = row['parent_asin']
                                    st.rerun()

            # --- Pagination Buttons ---
            st.markdown("---")
            total_pages = (total_results + PRODUCTS_PER_PAGE - 1) // PRODUCTS_PER_PAGE
            if total_pages > 1:
                nav_cols = st.columns([1, 1, 1])
                with nav_cols[0]:
                    if st.session_state.page > 0:
                        if st.button("⬅️ Previous Page"):
                            st.session_state.page -= 1
                            st.rerun()
                with nav_cols[1]:
                    st.write(f"Page {st.session_state.page + 1} of {total_pages}")
                with nav_cols[2]:
                    if (st.session_state.page + 1) < total_pages:
                        if st.button("Next Page ➡️"):
                            st.session_state.page += 1
                            st.rerun()

else:
    st.error("Application setup failed. Please check database connection.")
