import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
from wordcloud import WordCloud, STOPWORDS
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

@st.cache_data
def get_discrepancy_data(_conn, asin):
    """
    Fetches all necessary data for the discrepancy plot for a given product.
    This is cached per product view.
    """
    query = "SELECT review_id, rating, text_polarity, sentiment FROM discrepancy_data WHERE parent_asin = ?"
    df = pd.read_sql(query, _conn, params=(asin,))
    if not df.empty:
        df['discrepancy'] = (df['text_polarity'] - ((df['rating'] - 3.0) / 2.0)).abs()
        # --- NEW: Add jitter to separate overlapping points ---
        jitter_strength_rating = 0.1
        jitter_strength_polarity = 0.02
        df['rating_jittered'] = df['rating'] + np.random.uniform(-jitter_strength_rating, jitter_strength_rating, size=len(df))
        df['text_polarity_jittered'] = df['text_polarity'] + np.random.uniform(-jitter_strength_polarity, jitter_strength_polarity, size=len(df))
    return df
    
st.cache_data
def get_rating_distribution_data(_conn, asin):
    """Fetches the pre-computed rating distribution for a product."""
    return pd.read_sql("SELECT `1_star`, `2_star`, `3_star`, `4_star`, `5_star` FROM rating_distribution WHERE parent_asin = ?", _conn, params=(asin,))

def get_single_review(_conn, review_id):
    """Fetches a single review as a DataFrame."""
    return pd.read_sql("SELECT review_id, rating, sentiment, text FROM reviews WHERE review_id = ?", _conn, params=(review_id,))

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
# --- NEW: Session state for unified drill-down display ---
if 'drilldown_reviews' not in st.session_state: st.session_state.drilldown_reviews = pd.DataFrame()
if 'drilldown_title' not in st.session_state: st.session_state.drilldown_title = ""
if 'drilldown_page' not in st.session_state: st.session_state.drilldown_page = 1
if 'drilldown_rating_filter' not in st.session_state: st.session_state.drilldown_rating_filter = None
if 'drilldown_rating' not in st.session_state: st.session_state.drilldown_rating = None



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
            image_urls_str = product_details.get('image_urls')
            image_urls = image_urls_str.split(',') if pd.notna(image_urls_str) and image_urls_str else []
            
            thumbnail_url = image_urls[0] if image_urls else PLACEHOLDER_IMAGE_URL
            st.image(thumbnail_url, use_container_width=True)

            if image_urls:
                with st.popover("View Image Gallery"):
                    # Ensure index is not out of bounds if the product changes
                    if st.session_state.image_index >= len(image_urls):
                        st.session_state.image_index = 0

                    def next_image():
                        st.session_state.image_index = (st.session_state.image_index + 1) % len(image_urls)
                    
                    def prev_image():
                        st.session_state.image_index = (st.session_state.image_index - 1 + len(image_urls)) % len(image_urls)

                    st.image(image_urls[st.session_state.image_index], use_container_width=True)

                    if len(image_urls) > 1:
                        g_col1, g_col2, g_col3 = st.columns([1, 8, 1])
                        g_col1.button("⬅️", on_click=prev_image, use_container_width=True, key="gallery_prev")
                        g_col2.caption(f"Image {st.session_state.image_index + 1} of {len(image_urls)}")
                        g_col3.button("➡️", on_click=next_image, use_container_width=True, key="gallery_next")
            
        with right_col:
            # ... (Header stats remain the same) ...
            pass
            st.header(product_details['product_title'])
            st.caption(f"Category: {product_details['category']}")
            
            stat_cols = st.columns(2)
            avg_rating = product_details.get('average_rating', 0)
            review_count = product_details.get('review_count', 0)
            stat_cols[0].metric("Average Rating", f"{avg_rating:.2f} ⭐")
            stat_cols[1].metric("Total Reviews", f"{int(review_count):,}")

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
                    dist_data.columns = ['Star_Rating', 'Count']
                    dist_data['Star_Rating'] = dist_data['Star_Rating'].str.replace('_', ' ')
                    
                    sort_order = ['1 star', '2 star', '3 star', '4 star', '5 star']
                    
                    selection = alt.selection_point(fields=['Star_Rating'], empty=True, on='click', name="rating_selector")
                    color = alt.condition(selection, alt.value('orange'), alt.value('steelblue'))

                    chart = alt.Chart(dist_data).mark_bar().encode(
                        x=alt.X('Star_Rating:N', sort=sort_order, title="Stars"),
                        y=alt.Y('Count:Q', title="Number of Reviews"),
                        color=color,
                        tooltip=['Star_Rating', 'Count']
                    ).add_params(
                        selection
                    ).properties(title="Overall Rating Distribution")
                    
                    event = st.altair_chart(chart, use_container_width=True, on_select="rerun")

                    # Replace the old event handling block with this one
                    if event.selection and "rating_selector" in event.selection and event.selection["rating_selector"]:
                        selected_data_list = event.selection["rating_selector"]
                        if selected_data_list:
                            selected_rating_str = selected_data_list[0]['Star_Rating']
                            selected_rating_int = int(re.search(r'\d+', selected_rating_str).group())
                    
                            # If we clicked a new bar, reset the page and set the new filter
                            if st.session_state.drilldown_rating_filter != selected_rating_int:
                                st.session_state.drilldown_rating_filter = selected_rating_int
                                st.session_state.drilldown_page = 1 # Reset to page 1 for new selection
                    
                else:
                    st.warning("No rating distribution data available.")            
            with col2:
                st.markdown("#### Rating vs. Text Discrepancy")
                if not discrepancy_df.empty:
                    st.info("Hover over points to see details. Clicking points is not enabled in this version.")
                    # --- MODIFIED: Use jittered data for plotting and show original data on hover ---
                    plot = px.scatter(
                        discrepancy_df,
                        x="rating_jittered",
                        y="text_polarity_jittered",
                        color="discrepancy",
                        color_continuous_scale=px.colors.sequential.Viridis,
                        custom_data=['review_id'],
                        hover_name='review_id',
                        hover_data={
                            'rating': True, # Show original rating
                            'text_polarity': ':.2f', # Show original polarity, formatted
                            'discrepancy': ':.2f',
                            'rating_jittered': False, # Hide jittered value from hover
                            'text_polarity_jittered': False # Hide jittered value from hover
                        }
                    )
                    plot.update_xaxes(title_text='Rating')
                    plot.update_yaxes(title_text='Text Sentiment Polarity')

                    selected_point = plotly_events(plot, click_event=True, key="discrepancy_click")
                    
                    # --- CORRECTED: Robust event handling using pointIndex ---
                    if selected_point:
                        # The event returns a list of dicts, get the first one.
                        point_data = selected_point[0]
                        
                        # Check if the 'pointIndex' key exists
                        if 'pointIndex' in point_data:
                            # Use the index to look up the review_id in our original DataFrame
                            clicked_index = point_data['pointIndex']
                            clicked_review_id = discrepancy_df.iloc[clicked_index]['review_id']
                            
                            review_text = get_single_review_text(conn, clicked_review_id)
                            with st.expander(f"Full text for review: {clicked_review_id}", expanded=True):
                                st.markdown(f"> {review_text}")
                        else:
                            st.warning("Could not retrieve review details from the clicked point. Please try again.")
                        
                else:
                    st.warning("No discrepancy data available.")

            # Replace the old unified display block with this one
            # --- UNIFIED DRILL-DOWN DISPLAY ---
            if st.session_state.drilldown_rating_filter is not None:
                st.markdown("---")
                st.subheader(f"Displaying {st.session_state.drilldown_rating_filter}-Star Reviews (Page {st.session_state.drilldown_page})")
            
                # Fetch the reviews for the current page and selected rating
                drilldown_reviews = get_paginated_reviews(
                    conn, 
                    selected_asin, 
                    st.session_state.drilldown_page, 
                    REVIEWS_PER_PAGE, 
                    rating_filter=st.session_state.drilldown_rating_filter
                )
            
                if not drilldown_reviews.empty:
                    for index, row in drilldown_reviews.iterrows():
                        with st.container(border=True):
                            st.markdown(f"**Rating: {row['rating']} ⭐ | Sentiment: {row['sentiment']}**")
                            st.markdown(f"> {row['text']}")
            
                    # --- Pagination for Drill-Down ---
                    nav_cols = st.columns([1, 1, 1])
                    with nav_cols[0]:
                        if st.session_state.drilldown_page > 1:
                            if st.button("⬅️ Previous 5", key="drilldown_prev"):
                                st.session_state.drilldown_page -= 1
                                st.rerun()
            
                    nav_cols[1].write("") # Spacer
            
                    with nav_cols[2]:
                        # Show "Next" button only if we received a full page of results
                        if len(drilldown_reviews) == REVIEWS_PER_PAGE:
                            if st.button("Next 5 ➡️", key="drilldown_next"):
                                st.session_state.drilldown_page += 1
                                st.rerun()
                else:
                    st.info("No more reviews to display for this rating.")

        with reviews_tab:
            # ... (General review pagination logic remains the same) ...
            pass

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
