import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import altair as alt
from streamlit_plotly_events import plotly_events
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
from datetime import datetime
from st_aggrid import AgGrid, GridOptionsBuilder

# --- Configure logging ---
logging.basicConfig(level=logging.INFO)

# --- Altair Data Transformer ---
alt.data_transformers.enable('default', max_rows=None)

# --- App Configuration ---
# This points to the dataset and file that we have verified are correct.
KAGGLE_DATASET_SLUG = "wathiqsoualhi/amazon-3mcauley" 
DATABASE_PATH = "amazon_reviews_v5.db"  # Using v5 to ensure no caching issues
DATA_VERSION = 4                             # Matching the DB version

VERSION_FILE_PATH = ".db_version"
PRODUCTS_PER_PAGE = 16
REVIEWS_PER_PAGE = 5
PLACEHOLDER_IMAGE_URL = "https://via.placeholder.com/200"

# --- Data Loading & Caching Functions ---

def download_data_with_versioning(dataset_slug, db_path, version_path, expected_version):
    """Downloads data using the official Kaggle API library and handles authentication."""
    current_version = 4
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
    """Connects to the SQLite database."""
    try:
        return sqlite3.connect(path, uri=True, check_same_thread=False, timeout=15)
    except Exception as e:
        st.error(f"FATAL: Could not connect to database at '{path}'. Error: {e}")
        st.stop()

@st.cache_data
def get_all_categories(_conn):
    """Fetches a list of all unique categories."""
    df = pd.read_sql("SELECT DISTINCT category FROM products", _conn)
    categories = sorted(df['category'].dropna().unique().tolist())
    categories.insert(0, "--- Select a Category ---")
    return categories

def get_filtered_products(_conn, category, search_term, sort_by, limit, offset):
    """
    (CORRECTED) Fetches a paginated and filtered list of products directly from the database.
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
    """Fetches details for only one product."""
    return pd.read_sql("SELECT * FROM products WHERE parent_asin = ?", _conn, params=(asin,))
        
def get_single_review_text(conn, review_id):
    """Fetches the full text of a single review by its unique ID."""
    result = conn.execute("SELECT text FROM reviews WHERE review_id = ?", (review_id,)).fetchone()
    return result[0] if result else "Review text not found."

# This new function replaces get_discrepancy_data and get_rating_distribution_data
@st.cache_data
def get_filtered_data_for_product(_conn, asin, rating_filter, sentiment_filter, date_range):
    """
    Fetches all necessary data for the detail page, applying all filters directly in SQL.
    This is the core of the on-the-fly analysis.
    """
    query = "SELECT review_id, rating, text_polarity, sentiment, date FROM discrepancy_data WHERE parent_asin = ?"
    params = [asin]

    # Dynamically add conditions for each filter if it's being used
    if rating_filter:
        query += f" AND rating IN ({','.join('?' for _ in rating_filter)})"
        params.extend(rating_filter)

    if sentiment_filter:
        query += f" AND sentiment IN ({','.join('?' for _ in sentiment_filter)})"
        params.extend(sentiment_filter)

    if date_range and len(date_range) == 2:
        # Ensure the date format matches what's in the database ('YYYY-MM-DD')
        start_date = date_range[0].strftime('%Y-%m-%d')
        end_date = date_range[1].strftime('%Y-%m-%d')
        query += " AND date BETWEEN ? AND ?"
        params.extend([start_date, end_date])

    df = pd.read_sql(query, _conn, params=params)

    # Calculate discrepancy and jitter on the filtered data
    if not df.empty:
        df['discrepancy'] = (df['text_polarity'] - ((df['rating'] - 3.0) / 2.0)).abs()
        df['rating_jittered'] = df['rating'] + np.random.uniform(-0.1, 0.1, size=len(df))
        df['text_polarity_jittered'] = df['text_polarity'] + np.random.uniform(-0.02, 0.02, size=len(df))

    return df
    
# --- Main App ---
st.set_page_config(layout="wide", page_title="Amazon Review Explorer")
st.title("⚡ Amazon Reviews - Sentiment Dashboard")

download_data_with_versioning(KAGGLE_DATASET_SLUG, DATABASE_PATH, VERSION_FILE_PATH, DATA_VERSION)
conn = connect_to_db(DATABASE_PATH)

# Initialize session state for all filters and pages
if 'page' not in st.session_state: st.session_state.page = 0
if 'review_page' not in st.session_state: st.session_state.review_page = 1
if 'selected_product' not in st.session_state: st.session_state.selected_product = None
if 'category' not in st.session_state: st.session_state.category = "--- Select a Category ---"
if 'search_term' not in st.session_state: st.session_state.search_term = ""
if 'sort_by' not in st.session_state: st.session_state.sort_by = "Popularity (Most Reviews)"
if 'image_index' not in st.session_state: st.session_state.image_index = 0
if 'drilldown_rating' not in st.session_state: st.session_state.drilldown_rating = None
if 'discrepancy_review_id' not in st.session_state: st.session_state.discrepancy_review_id = None
# Add these two lines for the new reviews tab state
if 'all_reviews_page' not in st.session_state: st.session_state.all_reviews_page = 0
if 'all_reviews_sort' not in st.session_state: st.session_state.all_reviews_sort = "Newest First"
if 'filtered_review_ids' not in st.session_state: st.session_state.filtered_review_ids = None
if 'loaded_reviews_df' not in st.session_state: st.session_state.loaded_reviews_df = pd.DataFrame()

if conn:
    # --- DETAILED PRODUCT VIEW ---
    if st.session_state.selected_product:
        
        selected_asin = st.session_state.selected_product
    
        # --- AGGRESSIVE CACHING FOR ALL DATA FUNCTIONS ---
        # By caching all data-loading functions, we ensure that Streamlit only
        # re-runs the specific function whose inputs have changed.
    
        @st.cache_data
        def get_product_details(_conn, asin):
            """Fetches and caches the main details for the selected product."""
            return pd.read_sql("SELECT * FROM products WHERE parent_asin = ?", _conn, params=(asin,))
    
        @st.cache_data
        def get_product_date_range(_conn, asin):
            """Gets the min/max review dates for the date picker."""
            return _conn.execute(
                "SELECT MIN(date), MAX(date) FROM reviews WHERE parent_asin=?", (asin,)
            ).fetchone()
    
        @st.cache_data(show_spinner="Analyzing sentiment data...")
        def get_filtered_data_for_charts(_conn, asin, rating_filter, sentiment_filter, date_range_tuple):
            """
            (HEAVILY CACHED) Fetches and processes data for the charts on the first tab.
            This now only runs when the filters in the sidebar are changed.
            """
            query = "SELECT review_id, rating, text_polarity, sentiment, date FROM discrepancy_data WHERE parent_asin = ?"
            params = [asin]
            if rating_filter:
                query += f" AND rating IN ({','.join('?' for _ in rating_filter)})"
                params.extend(rating_filter)
            if sentiment_filter:
                query += f" AND sentiment IN ({','.join('?' for _ in sentiment_filter)})"
                params.extend(sentiment_filter)
            if date_range_tuple and len(date_range_tuple) == 2:
                start_date, end_date = date_range_tuple
                query += " AND date BETWEEN ? AND ?"
                params.extend([start_date, end_date])
            
            df = pd.read_sql(query, _conn, params=params)
            if not df.empty:
                df['discrepancy'] = (df['text_polarity'] - ((df['rating'] - 3.0) / 2.0)).abs()
                df['rating_jittered'] = df['rating'] + np.random.uniform(-0.1, 0.1, size=len(df))
                df['text_polarity_jittered'] = df['text_polarity'] + np.random.uniform(-0.02, 0.02, size=len(df))
            return df
    
        @st.cache_data(show_spinner=False)
        def get_single_review_by_id(_conn, review_id):
            """Gets the text for a single review, used for the discrepancy plot click."""
            result = _conn.execute("SELECT text FROM reviews WHERE review_id = ?", (review_id,)).fetchone()
            return result[0] if result else "Review text not found."
        
        # --- This is the new, efficient, no-pandas pagination function ---
        @st.cache_data(show_spinner="Fetching reviews...")
        def get_reviews_for_page_raw(_conn, asin, page_num):
            """
            (NO PANDAS) Fetches a single page of raw review data directly from the DB.
            This is the most memory-efficient method. Caching makes it instant on revisit.
            """
            offset = (page_num - 1) * 25
            cursor = _conn.cursor()
            cursor.execute(
                "SELECT rating, sentiment, date, text FROM reviews WHERE parent_asin = ? ORDER BY date DESC LIMIT 25 OFFSET ?",
                (asin, offset)
            )
            return cursor.fetchall()
        
        @st.cache_data(show_spinner=False)
        def get_total_review_count(_conn, asin):
            """Gets the total number of reviews for pagination controls."""
            count_df = pd.read_sql("SELECT review_count FROM products WHERE parent_asin = ?", _conn, params=(asin,))
            return count_df['review_count'].iloc[0]
    
        # --- RENDER THE PAGE ---
        
        product_details_df = get_product_details(conn, selected_asin)
        if product_details_df.empty:
            st.error("Product details could not be found.")
            st.stop()
        product_details = product_details_df.iloc[0]
    
        if st.button("⬅️ Back to Search"):
            # Clear state related to the detail view to ensure a clean slate
            for key in ['selected_product', 'review_page', 'image_index', 'drilldown_rating', 'discrepancy_review_id', 'all_reviews_page', 'current_product_asin']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
    
        # --- Sidebar Filters ---
        st.sidebar.header("Interactive Filters")
        min_date_db, max_date_db = get_product_date_range(conn, selected_asin)
        min_date = datetime.strptime(min_date_db, '%Y-%m-%d').date() if min_date_db else datetime(2000, 1, 1).date()
        max_date = datetime.strptime(max_date_db, '%Y-%m-%d').date() if max_date_db else datetime.now().date()
        
        selected_date_range = st.sidebar.date_input("Filter by Date Range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
        selected_ratings = st.sidebar.multiselect("Filter by Star Rating", options=[1, 2, 3, 4, 5], default=[1, 2, 3, 4, 5])
        selected_sentiments = st.sidebar.multiselect("Filter by Sentiment", options=['Positive', 'Negative', 'Neutral'], default=['Positive', 'Negative', 'Neutral'])
    
        # --- Header and Image Gallery ---
        # (This section of your code is fine and does not need changes)
        left_col, right_col = st.columns([1, 2])
        with left_col:
            # ... your existing image gallery code ...
            image_urls_str = product_details.get('image_urls')
            image_urls = image_urls_str.split(',') if pd.notna(image_urls_str) and image_urls_str else []
            thumbnail_url = image_urls[0] if image_urls else PLACEHOLDER_IMAGE_URL
            st.image(thumbnail_url, use_container_width=True)
        with right_col:
            st.header(product_details['product_title'])
            st.caption(f"Category: {product_details['category']}")
            stat_cols = st.columns(2)
            stat_cols[0].metric("Average Rating", f"{product_details.get('average_rating', 0):.2f} ⭐")
            stat_cols[1].metric("Total Reviews", f"{int(product_details.get('review_count', 0)):,}")
    
        st.markdown("---")
    
        # --- Main Content Tabs ---
        vis_tab, reviews_tab = st.tabs(["📊 Sentiment Analysis", "💬 Individual Reviews"])
    
        with vis_tab:
            # Convert date range to a tuple to make it hashable for caching
            date_tuple = (selected_date_range[0].strftime('%Y-%m-%d'), selected_date_range[1].strftime('%Y-%m-%d'))
            
            # This function is now heavily cached and will only re-run if filters change.
            filtered_data = get_filtered_data_for_charts(conn, selected_asin, tuple(selected_ratings), tuple(selected_sentiments), date_tuple)
            
            st.subheader("Live Analysis on Filtered Data")
            st.write(f"Displaying analysis for **{len(filtered_data)}** reviews matching your criteria.")
    
            if filtered_data.empty:
                st.warning("No reviews match the selected filters.")
            else:
                col1, col2 = st.columns(2)
                with col1:
                    # ... (Your rating distribution chart code is fine here) ...
                    st.markdown("#### Rating Distribution (Live)")
                    rating_counts_df = filtered_data['rating'].value_counts().sort_index().reset_index()
                    rating_counts_df.columns = ['Rating', 'Count']
                    chart = alt.Chart(rating_counts_df).mark_bar().encode(x='Rating:O', y='Count:Q').properties(title="Filtered Rating Distribution")
                    st.altair_chart(chart, use_container_width=True)
                with col2:
                    # ... (Your discrepancy plot code is fine here) ...
                    st.markdown("#### Rating vs. Text Discrepancy (Live & Interactive)")
                    plot = px.scatter(filtered_data, x="rating_jittered", y="text_polarity_jittered", color="discrepancy", custom_data=['review_id'])
                    selected_point = plotly_events(plot, click_event=True, key="discrepancy_click")
                    if selected_point:
                        review_id = selected_point[0]['customdata'][0]
                        st.session_state.discrepancy_review_id = review_id
            
            if st.session_state.get('discrepancy_review_id'):
                st.markdown("---")
                st.subheader(f"Selected Review: {st.session_state.discrepancy_review_id}")
                review_text = get_single_review_by_id(conn, st.session_state.discrepancy_review_id)
                with st.container(border=True):
                    st.markdown(f"> {review_text}")
                if st.button("Close Review Snippet"):
                    del st.session_state.discrepancy_review_id
                    st.rerun()
    
        with reviews_tab:
            # This tab is now completely independent and uses the ultra-lightweight raw SQL method.
            # It no longer triggers chart recalculations.
            if 'all_reviews_page' not in st.session_state:
                st.session_state.all_reviews_page = 1
            if 'current_product_asin' not in st.session_state or st.session_state.current_product_asin != selected_asin:
                st.session_state.current_product_asin = selected_asin
                st.session_state.all_reviews_page = 1
    
            total_reviews = get_total_review_count(conn, selected_asin)
            total_pages = (total_reviews + 24) // 25
            
            reviews_on_page = get_reviews_for_page_raw(conn, selected_asin, st.session_state.all_reviews_page)
            
            st.markdown("---")
            if reviews_on_page:
                for row in reviews_on_page:
                    rating, sentiment, date, text = row
                    sentiment_color = "green" if sentiment == 'Positive' else "red" if sentiment == 'Negative' else "orange"
                    with st.container(border=True):
                        st.markdown(f"**Rating: :{sentiment_color}[{rating} ⭐]** | **Date:** {date}")
                        st.markdown(f"> {text}")
            else:
                 st.warning("No reviews were found for this product.")
            st.markdown("---")
    
            # Pagination Controls
            nav_cols = st.columns([1, 1, 1])
            with nav_cols[0]:
                if st.session_state.all_reviews_page > 1:
                    if st.button("⬅️ Previous", use_container_width=True, key="prev_reviews"):
                        st.session_state.all_reviews_page -= 1
                        st.rerun()
            with nav_cols[1]:
                if total_pages > 0:
                    st.write(f"Page **{st.session_state.all_reviews_page}** of **{total_pages}**")
            with nav_cols[2]:
                if st.session_state.all_reviews_page < total_pages:
                    if st.button("Next ➡️", use_container_width=True, key="next_reviews"):
                        st.session_state.all_reviews_page += 1
                        st.rerun()                
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
            def on_category_change():
                st.session_state.page = 0
            st.session_state.category = st.selectbox("Filter by Category", available_categories, index=available_categories.index(st.session_state.category), on_change=on_category_change)
        with col3:
            st.session_state.sort_by = st.selectbox("Sort By", ["Popularity (Most Reviews)", "Highest Rating", "Lowest Rating"], index=["Popularity (Most Reviews)", "Highest Rating", "Lowest Rating"].index(st.session_state.sort_by))

        if st.session_state.category == "--- Select a Category ---":
            st.info("Please select a category to view products.")
        else:
            paginated_results, total_results = get_filtered_products(
                conn, st.session_state.category, st.session_state.search_term, st.session_state.sort_by, 
                limit=PRODUCTS_PER_PAGE, 
                offset=st.session_state.page * PRODUCTS_PER_PAGE
            )
    
            st.markdown("---")
            st.header(f"Found {total_results} Products in '{st.session_state.category}'")
            
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
                    pass
    
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
