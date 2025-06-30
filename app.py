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
DATA_VERSION = 1                             # Matching the DB version

VERSION_FILE_PATH = ".db_version"
PRODUCTS_PER_PAGE = 16
REVIEWS_PER_PAGE = 5
PLACEHOLDER_IMAGE_URL = "https://via.placeholder.com/200"

# --- Data Loading Function ---
def download_data_with_versioning(dataset_slug, db_path, version_path, expected_version):
    """Downloads data using the official Kaggle API library and handles authentication."""
    current_version = 1
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
    categories.insert(0, "--- Select a Category ---")
    return categories

def get_filtered_products(_conn, category, search_term, sort_by, limit, offset):
    """
    Fetches a paginated and filtered list of products directly from the database.
    This is highly memory-efficient.
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

    total_count = _conn.execute(count_query, tuple(params)).fetchone()[0]
    query += f" LIMIT ? OFFSET ?"
    params.extend([limit, offset])
    
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

def get_paginated_reviews(_conn, asin, page_num, page_size, rating_filter=None):
    """
    Fetches a small 'page' of raw reviews to display, with an optional rating filter.
    """
    offset = (page_num - 1) * page_size
    params = [asin]
    query = f"SELECT rating, sentiment, text FROM reviews WHERE parent_asin = ?"

    if rating_filter is not None:
        query += " AND rating = ?"
        params.append(rating_filter)

    query += f" LIMIT ? OFFSET ?"
    params.extend([page_size, offset])
    
    return pd.read_sql(query, _conn, params=params)


# --- Main App ---
st.set_page_config(layout="wide", page_title="Amazon Review Explorer")
st.title("⚡ Amazon Reviews - Sentiment Dashboard")

REQUIRED_TABLES = ['products', 'reviews', 'discrepancy_data', 'rating_distribution']

download_data_with_versioning(KAGGLE_DATASET_SLUG, DATABASE_PATH, VERSION_FILE_PATH, DATA_VERSION)
conn = connect_to_db(DATABASE_PATH, REQUIRED_TABLES)

# Initialize session state for all filters and pages
if 'page' not in st.session_state: st.session_state.page = 0
if 'review_page' not in st.session_state: st.session_state.review_page = 1
if 'selected_product' not in st.session_state: st.session_state.selected_product = None
if 'category' not in st.session_state: st.session_state.category = "--- Select a Category ---"
if 'search_term' not in st.session_state: st.session_state.search_term = ""
if 'sort_by' not in st.session_state: st.session_state.sort_by = "Popularity (Most Reviews)"
if 'image_index' not in st.session_state: st.session_state.image_index = 0
if 'drilldown_rating' not in st.session_state: st.session_state.drilldown_rating = None
if 'drilldown_page' not in st.session_state: st.session_state.drilldown_page = 1


if conn:
    # --- DETAILED PRODUCT VIEW ---
    if st.session_state.selected_product:
        selected_asin = st.session_state.selected_product
        product_details_df = get_single_product_details(conn, selected_asin)
        
        if product_details_df.empty:
            st.error("Product details could not be found.")
            st.stop()
        
        product_details = product_details_df.iloc[0]

        if st.button("⬅️ Back to Search"):
            st.session_state.selected_product = None
            st.session_state.review_page = 1
            st.session_state.image_index = 0
            st.session_state.drilldown_rating = None
            st.session_state.drilldown_page = 1
            st.rerun()

        # Header Layout with Popover Image Gallery
        left_col, right_col = st.columns([1, 2])
        with left_col:
            # ... (Image gallery code remains the same) ...
            pass
        with right_col:
            # ... (Header stats remain the same) ...
            pass

        st.markdown("---")
        
        # --- Visualization Tabs ---
        vis_tab, reviews_tab = st.tabs(["📊 Sentiment Analysis", "💬 Individual Reviews"])

        with vis_tab:
            st.subheader("Sentiment Analysis Visualizations")
            
            discrepancy_df = get_discrepancy_data(conn, selected_asin)
            rating_dist_df = get_rating_distribution_data(conn, selected_asin)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Rating Distribution (Click a bar to see reviews)")
                if not rating_dist_df.empty:
                    dist_data = rating_dist_df.T.reset_index()
                    dist_data.columns = ['Star Rating', 'Count']
                    dist_data['Star Rating'] = dist_data['Star Rating'].str.replace('_', ' ')
                    
                    # --- MODIFIED: Interactive Altair Chart ---
                    selection = alt.selection_point(fields=['Star Rating'], empty=True, on='click')
                    color = alt.condition(selection, alt.value('orange'), alt.value('steelblue'))

                    chart = alt.Chart(dist_data).mark_bar().encode(
                        x=alt.X('Star Rating', sort=None, title="Stars"),
                        y=alt.Y('Count', title="Number of Reviews"),
                        color=color,
                        tooltip=['Star Rating', 'Count']
                    ).add_params(
                        selection
                    ).properties(title="Overall Rating Distribution")
                    
                    event = st.altair_chart(chart, use_container_width=True, on_select="rerun")

                    # Handle drill-down from chart selection
                    if event.selection and event.selection["Star Rating"]:
                        selected_rating_str = event.selection["Star Rating"][0]
                        # Convert "5 star" string to integer 5
                        selected_rating_int = int(selected_rating_str.split(' ')[0])
                        
                        # If a new bar is clicked, reset the page
                        if st.session_state.drilldown_rating != selected_rating_int:
                            st.session_state.drilldown_rating = selected_rating_int
                            st.session_state.drilldown_page = 1
                    
                else:
                    st.warning("No rating distribution data available.")
            
            with col2:
                st.markdown("#### Rating vs. Text Discrepancy")
                if not discrepancy_df.empty:
                    st.info("Hover over points to see details. Clicking points is not enabled in this version.")
                    plot = px.scatter(discrepancy_df, x="rating", y="text_polarity", color="discrepancy", title="Discrepancy Plot")
                    st.plotly_chart(plot, use_container_width=True)
                else:
                    st.warning("No discrepancy data available.")

            # --- NEW: Drill-down review display section ---
            if st.session_state.drilldown_rating:
                st.markdown("---")
                st.subheader(f"Displaying {st.session_state.drilldown_rating}-Star Reviews")
                
                drilldown_reviews = get_paginated_reviews(
                    conn, selected_asin, 
                    st.session_state.drilldown_page, 
                    REVIEWS_PER_PAGE, 
                    rating_filter=st.session_state.drilldown_rating
                )

                if not drilldown_reviews.empty:
                    for index, row in drilldown_reviews.iterrows():
                        st.markdown(f"> {row['text']}")
                        st.divider()
                    
                    if len(drilldown_reviews) == REVIEWS_PER_PAGE:
                        if st.button("Load More Reviews"):
                            st.session_state.drilldown_page += 1
                            st.rerun()
                else:
                    st.info("No more reviews to display for this rating.")


        with reviews_tab:
            # ... (General review pagination logic remains the same) ...
            pass

    # --- MAIN SEARCH PAGE ---
    else:
        # ... (Main page logic remains the same) ...
        pass

else:
    st.error("Application setup failed. Please check database connection.")
