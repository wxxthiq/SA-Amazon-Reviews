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
        # This will only trigger if the download fails to place the file.
        st.error(f"Database file not found at path: {path}. Please ensure the download was successful.")
        st.stop()

    try:
        conn = sqlite3.connect(path, uri=True, check_same_thread=False, timeout=15)
        cursor = conn.cursor()
        
        # Verify that the necessary tables exist before proceeding.
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
def get_product_summary_data(_conn):
    """Fetches the main product gallery data. This is cached as it's large and loaded once."""
    return pd.read_sql("SELECT * FROM products", _conn)

@st.cache_data
def get_all_categories(_conn):
    """Fetches a list of all unique categories. Cached for efficiency on the main page."""
    df = pd.read_sql("SELECT DISTINCT category FROM products", _conn)
    categories = sorted(df['category'].dropna().unique().tolist())
    categories.insert(0, "All") # Add 'All' as the default option
    return categories

# --- MODIFIED: Caching removed from detail-page functions to prevent memory leaks ---
def get_discrepancy_data(_conn, asin):
    """
    Fetches the lightweight data for the discrepancy plot.
    Not cached to prevent memory accumulation across multiple product views.
    """
    df = pd.read_sql("SELECT rating, text_polarity FROM discrepancy_data WHERE parent_asin = ?", _conn, params=(asin,))
    if not df.empty:
        df['discrepancy'] = (df['text_polarity'] - ((df['rating'] - 3.0) / 2.0)).abs()
    return df

# --- MODIFIED: Caching removed ---
def get_rating_distribution_data(_conn, asin):
    """
    Fetches the pre-computed rating distribution for a product.
    Not cached to prevent memory accumulation.
    """
    return pd.read_sql("SELECT `1_star`, `2_star`, `3_star`, `4_star`, `5_star` FROM rating_distribution WHERE parent_asin = ?", _conn, params=(asin,))

def get_paginated_reviews(_conn, asin, page_num, page_size):
    """Fetches a small 'page' of raw reviews to display."""
    offset = (page_num - 1) * page_size
    return pd.read_sql(f"SELECT rating, sentiment, text FROM reviews WHERE parent_asin = ? LIMIT ? OFFSET ?", _conn, params=(asin, page_size, offset))


# --- Main App ---
st.set_page_config(layout="wide", page_title="Amazon Review Explorer")
st.title("⚡ Amazon Reviews - Sentiment Dashboard")

# Define the tables this "lite" version of the app requires
REQUIRED_TABLES = ['products', 'reviews', 'discrepancy_data', 'rating_distribution']

download_data_with_versioning(KAGGLE_DATASET_SLUG, DATABASE_PATH, VERSION_FILE_PATH, DATA_VERSION)
conn = connect_to_db(DATABASE_PATH, REQUIRED_TABLES)

# Initialize session state
if 'page' not in st.session_state: st.session_state.page = 0
if 'review_page' not in st.session_state: st.session_state.review_page = 1
if 'selected_product' not in st.session_state: st.session_state.selected_product = None
if 'search_term' not in st.session_state: st.session_state.search_term = ""
if 'category' not in st.session_state: st.session_state.category = "All"
if 'sort_by' not in st.session_state: st.session_state.sort_by = "Popularity (Most Reviews)"


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
        # Image Carousel Logic
        image_urls_str = product_details.get('image_urls')
        image_urls = image_urls_str.split(',') if pd.notna(image_urls_str) else []
        if image_urls:
            st.image(image_urls[0], use_container_width=True) # Display first image
        else:
            st.image(PLACEHOLDER_IMAGE_URL, use_container_width=True)


        st.markdown("---")
        
        # --- Visualization Tabs ---
        vis_tab, reviews_tab = st.tabs(["📊 Sentiment Analysis", "💬 Individual Reviews"])

        with vis_tab:
            st.subheader("Sentiment Analysis Visualizations")
            
            # These functions are now called live on each run for this page
            discrepancy_df = get_discrepancy_data(conn, selected_asin)
            rating_dist_df = get_rating_distribution_data(conn, selected_asin)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Rating Distribution")
                if not rating_dist_df.empty:
                    dist_data = rating_dist_df.T.reset_index()
                    dist_data.columns = ['Star Rating', 'Count']
                    dist_data['Star Rating'] = dist_data['Star Rating'].str.replace('_', ' ')
                    
                    chart = alt.Chart(dist_data).mark_bar().encode(
                        x=alt.X('Star Rating', sort=None, title="Stars"),
                        y=alt.Y('Count', title="Number of Reviews"),
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
                st.info("No more reviews to display for this product.")

    # --- MAIN SEARCH PAGE ---
    else:
        st.session_state.review_page = 1
        st.header("Search for Products")
        
        # --- Search and Filter Controls ---
        col1, col2, col3 = st.columns(3)
        with col1:
            st.session_state.search_term = st.text_input("Search by product title:", value=st.session_state.search_term)
        with col2:
            available_categories = get_all_categories(conn)
            st.session_state.category = st.selectbox("Filter by Category", available_categories, index=available_categories.index(st.session_state.category))
        with col3:
            st.session_state.sort_by = st.selectbox("Sort By", ["Popularity (Most Reviews)", "Highest Rating", "Lowest Rating"], index=["Popularity (Most Reviews)", "Highest Rating", "Lowest Rating"].index(st.session_state.sort_by))

        # --- Logic to display results only when a category is selected ---
        if st.session_state.category == "All":
            st.info("Please select a category to view products.")
        else:
            # Apply all filters to the main dataframe
            search_results_df = products_df
            if st.session_state.category != "All":
                search_results_df = search_results_df[search_results_df['category'] == st.session_state.category]
            if st.session_state.search_term:
                search_results_df = search_results_df[search_results_df['product_title'].str.contains(st.session_state.search_term, case=False, na=False)]
            
            # Apply sorting
            if st.session_state.sort_by == "Popularity (Most Reviews)":
                search_results_df = search_results_df.sort_values(by="review_count", ascending=False)
            elif st.session_state.sort_by == "Highest Rating":
                search_results_df = search_results_df.sort_values(by="average_rating", ascending=False)
            else: # Lowest Rating
                search_results_df = search_results_df.sort_values(by="average_rating", ascending=True)

            st.markdown("---")
            total_results = len(search_results_df)
            st.header(f"Found {total_results} Products in '{st.session_state.category}'")
            
            # --- Pagination Logic ---
            start_idx = st.session_state.page * PRODUCTS_PER_PAGE
            end_idx = start_idx + PRODUCTS_PER_PAGE
            paginated_results = search_results_df.iloc[start_idx:end_idx]

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
