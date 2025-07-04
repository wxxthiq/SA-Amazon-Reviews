import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
from wordcloud import WordCloud
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

    if rating_filter:
        query += f" AND rating IN ({','.join('?' for _ in rating_filter)})"
        params.extend(rating_filter)

    if sentiment_filter:
        query += f" AND sentiment IN ({','.join('?' for _ in sentiment_filter)})"
        params.extend(sentiment_filter)

    if date_range and len(date_range) == 2:
        start_date = date_range[0].strftime('%Y-%m-%d')
        end_date = date_range[1].strftime('%Y-%m-%d')
        query += " AND date BETWEEN ? AND ?"
        params.extend([start_date, end_date])

    df = pd.read_sql(query, _conn, params=params)

    if not df.empty:
        # --- STABLE JITTER CALCULATION ---
        # We use a fixed seed for the random number generator. This ensures that
        # the jitter is the same every time for the same filtered dataset.
        rng = np.random.default_rng(seed=42) # Using a fixed seed

        df['discrepancy'] = (df['text_polarity'] - ((df['rating'] - 3.0) / 2.0)).abs()
        df['rating_jittered'] = df['rating'] + rng.uniform(-0.1, 0.1, size=len(df))
        df['text_polarity_jittered'] = df['text_polarity'] + rng.uniform(-0.02, 0.02, size=len(df))

    return df
    
# --- Main App ---
st.set_page_config(layout="wide", page_title="Amazon Review Explorer")
st.title("⚡ Amazon Reviews - Sentiment Dashboard")

download_data_with_versioning(KAGGLE_DATASET_SLUG, DATABASE_PATH, VERSION_FILE_PATH, DATA_VERSION)
conn = connect_to_db(DATABASE_PATH)

# Initialize session state for all filters and pages
if 'page' not in st.session_state: st.session_state.page = 0
if 'selected_product' not in st.session_state: st.session_state.selected_product = None
if 'category' not in st.session_state: st.session_state.category = "--- Select a Category ---"
if 'search_term' not in st.session_state: st.session_state.search_term = ""
if 'sort_by' not in st.session_state: st.session_state.sort_by = "Popularity (Most Reviews)"
if 'image_index' not in st.session_state: st.session_state.image_index = 0
if 'drilldown_rating' not in st.session_state: st.session_state.drilldown_rating = None
if 'discrepancy_review_id' not in st.session_state: st.session_state.discrepancy_review_id = None

if conn:
    # --- DETAILED PRODUCT VIEW ---
    if st.session_state.selected_product:
        
        selected_asin = st.session_state.selected_product
    
        # --- TOP-LEVEL HELPER FUNCTIONS ---
        @st.cache_data
        def get_product_details(_conn, asin):
            return pd.read_sql("SELECT * FROM products WHERE parent_asin = ?", _conn, params=(asin,))
    
        @st.cache_data
        def get_product_date_range(_conn, asin):
            return _conn.execute("SELECT MIN(date), MAX(date) FROM reviews WHERE parent_asin=?", (asin,)).fetchone()
    
        # --- RENDER STATIC PAGE ELEMENTS (HEADER, SIDEBAR) ---
        product_details_df = get_product_details(conn, selected_asin)
        if product_details_df.empty:
            st.error("Product details could not be found.")
            st.stop()
        product_details = product_details_df.iloc[0]
    
        if st.button("⬅️ Back to Search"):
            # Clear all session state keys related to the detail view for a clean slate
            keys_to_clear = [k for k in st.session_state.keys() if k not in ['page', 'category', 'search_term', 'sort_by']]
            for key in keys_to_clear:
                del st.session_state[key]
            st.rerun()
    
        # (Image gallery and header)
        left_col, right_col = st.columns([1, 2])
        with left_col:
            image_urls_str = product_details.get('image_urls')
            image_urls = image_urls_str.split(',') if pd.notna(image_urls_str) else []
            st.image(image_urls[0] if image_urls else PLACEHOLDER_IMAGE_URL, use_container_width=True)
            # --- RESTORED IMAGE GALLERY POPOVER ---
            if image_urls:
                with st.popover("View Image Gallery"):
                    if 'image_index' not in st.session_state or st.session_state.image_index >= len(image_urls):
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
            st.header(product_details['product_title'])
            st.caption(f"Category: {product_details['category']}")
            st.metric("Average Rating", f"{product_details.get('average_rating', 0):.2f} ⭐")
            st.metric("Total Reviews", f"{int(product_details.get('review_count', 0)):,}")
    
        # --- SIDEBAR FILTERS (Now only control the 'Sentiment Analysis' tab) ---
        st.sidebar.header("Interactive Filters")
        min_date_db, max_date_db = get_product_date_range(conn, selected_asin)
        min_date = datetime.strptime(min_date_db, '%Y-%m-%d').date() if min_date_db else datetime(2000, 1, 1).date()
        max_date = datetime.strptime(max_date_db, '%Y-%m-%d').date() if max_date_db else datetime.now().date()
        
        default_date_range = (min_date, max_date)
        default_ratings = [1, 2, 3, 4, 5]
        default_sentiments = ['Positive', 'Negative', 'Neutral']
    
        def reset_all_filters():
            st.session_state.date_filter = default_date_range
            st.session_state.rating_filter = default_ratings
            st.session_state.sentiment_filter = default_sentiments
    
        selected_date_range = st.sidebar.date_input("Filter by Date Range", value=default_date_range, min_value=min_date, max_value=max_date, key='date_filter')
        selected_ratings = st.sidebar.multiselect("Filter by Star Rating", default_ratings, default=default_ratings, key='rating_filter')
        selected_sentiments = st.sidebar.multiselect("Filter by Sentiment", default_sentiments, default=default_sentiments, key='sentiment_filter')
        st.sidebar.button("Reset All Filters", on_click=reset_all_filters, use_container_width=True)
    
        # --- RENDER TABS ---
        vis_tab, wordcloud_tab, breakdown_tab  = st.tabs(["📊 Sentiment Analysis","☁️ Word Clouds", "Rating Breakdown"])
    
        # ======================== SENTIMENT ANALYSIS TAB ========================
        with vis_tab:
            
            # This function is LOCAL to this tab and RESPONDS to filters
            @st.cache_data
            def get_data_for_sentiment_charts(_conn, asin, rating_filter, sentiment_filter, date_range_tuple):
                query = "SELECT review_id, rating, sentiment, date, text_polarity FROM discrepancy_data WHERE parent_asin = ?"
                params = [asin]
                if rating_filter:
                    query += f" AND rating IN ({','.join('?' for _ in rating_filter)})"
                    params.extend(rating_filter)
                if sentiment_filter:
                    query += f" AND sentiment IN ({','.join('?' for _ in sentiment_filter)})"
                    params.extend(sentiment_filter)
                if date_range_tuple and len(date_range_tuple) == 2:
                    start_date, end_date = date_range_tuple
                    query += f" AND date BETWEEN ? AND ?"
                    params.extend(date_range_tuple)
                
                df = pd.read_sql(query, _conn, params=params)
                if not df.empty:
                    rng = np.random.default_rng(seed=42)
                    df['discrepancy'] = (df['text_polarity'] - ((df['rating'] - 3.0) / 2.0)).abs()
                    df['rating_jittered'] = df['rating'] + rng.uniform(-0.1, 0.1, size=len(df))
                    df['text_polarity_jittered'] = df['text_polarity'] + rng.uniform(-0.02, 0.02, size=len(df))
                return df
    
            st.subheader("Live Analysis on Filtered Data")
            date_tuple = (selected_date_range[0].strftime('%Y-%m-%d'), selected_date_range[1].strftime('%Y-%m-%d'))
            chart_data = get_data_for_sentiment_charts(conn, selected_asin, tuple(selected_ratings), tuple(selected_sentiments), date_tuple)
            
            st.write(f"Displaying analysis for **{len(chart_data)}** reviews matching your criteria.")
    
            if chart_data.empty:
                st.warning("No reviews match the selected filters.")
            else:
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("#### Rating Distribution (Live)")
                    rating_counts_df = chart_data['rating'].value_counts().sort_index().reset_index()
                    rating_counts_df.columns = ['Rating', 'Count']
                    chart = alt.Chart(rating_counts_df).mark_bar().encode(
                        x=alt.X('Rating:O', title="Stars"),
                        y=alt.Y('Count:Q', title="Number of Reviews"),
                        tooltip=['Rating', 'Count']
                    ).properties(title="Filtered Rating Distribution")
                    st.altair_chart(chart, use_container_width=True)
                                    # --- Sentiment Distribution Chart (NEW) ---
                    st.markdown("#### Sentiment Distribution (Live)")
                    sentiment_counts_df = chart_data['sentiment'].value_counts().reset_index()
                    sentiment_counts_df.columns = ['Sentiment', 'Count']
                    
                    sentiment_chart = alt.Chart(sentiment_counts_df).mark_bar().encode(
                        x=alt.X('Sentiment:N', title="Sentiment", sort='-y'),
                        y=alt.Y('Count:Q', title="Number of Reviews"),
                        color=alt.Color('Sentiment:N',
                                        scale=alt.Scale(
                                            domain=['Positive', 'Neutral', 'Negative'],
                                            range=['#1a9850', '#cccccc', '#d73027']
                                        ),
                                        legend=None),
                        tooltip=['Sentiment', 'Count']
                    ).properties(title="Filtered Sentiment Distribution")
                    st.altair_chart(sentiment_chart, use_container_width=True)
                #
                with col2:
                    st.markdown("#### Rating vs. Text Discrepancy (Live & Interactive)")
                    
                    # We use the 'chart_data' DataFrame which is already filtered and prepared
                    plot = px.scatter(
                        chart_data,
                        x="rating_jittered",
                        y="text_polarity_jittered",
                        color="discrepancy",
                        color_continuous_scale=px.colors.sequential.Viridis,
                        custom_data=['review_id'],
                        hover_name='review_id',
                        hover_data={
                            'rating': True, 'text_polarity': ':.2f', 'discrepancy': ':.2f',
                            'rating_jittered': False, 'text_polarity_jittered': False
                        }
                    )
                    plot.update_xaxes(title_text='Rating')
                    plot.update_yaxes(title_text='Text Sentiment Polarity')
                    selected_point = plotly_events(plot, click_event=True, key="discrepancy_click")
                    
                    # --- DEFINITIVE FIX FOR CLICK AND CLOSE LOGIC ---
    
                    # 1. Handle the "Close" button click first and exit immediately.
                    # This is the highest priority action.
                    if st.session_state.get('discrepancy_review_id'):
                        st.markdown("---")
                        st.subheader(f"Selected Review: {st.session_state.discrepancy_review_id}")
                        review_text = get_single_review_text(conn, st.session_state.discrepancy_review_id)
                        
                        with st.container(border=True):
                            st.markdown(f"> {review_text}")
                        
                        # If the close button is clicked, clear the state and force an immediate rerun.
                        if st.button("Close Review Snippet", key="close_snippet"):
                            st.session_state.discrepancy_review_id = None
                            st.rerun() # This is crucial to prevent the stale click event from being processed.
    
                    # 2. If the close button was NOT pressed, then process any new plot clicks.
                    if selected_point:
                        point_data = selected_point[0]
                        if 'pointIndex' in point_data:
                            clicked_index = point_data['pointIndex']
                            if clicked_index < len(chart_data):
                                review_id = chart_data.iloc[clicked_index]['review_id']
                                # Only rerun if it's a new review selection.
                                if st.session_state.get('discrepancy_review_id') != review_id:
                                    st.session_state.discrepancy_review_id = review_id
                                    st.rerun()
    
                st.markdown("---")
                time_df = chart_data.copy()
                time_df['date'] = pd.to_datetime(time_df['date'])
                time_df['month'] = time_df['date'].dt.to_period('M').dt.start_time
                col3, col4 = st.columns(2)
                with col3:
                    st.markdown("#### Rating Distribution Over Time")
                    rating_counts_over_time = time_df.groupby(['month', 'rating']).size().reset_index(name='count')
                    if not rating_counts_over_time.empty:
                        rating_stream_chart = px.area(
                            rating_counts_over_time, x='month', y='count', color='rating',
                            title="Volume of Reviews by Star Rating",
                            color_discrete_map={5: '#1a9850', 4: '#91cf60', 3: '#d9ef8b', 2: '#fee08b', 1: '#d73027'},
                            category_orders={"rating": [5, 4, 3, 2, 1]}
                        )
                        st.plotly_chart(rating_stream_chart, use_container_width=True)
                with col4:
                    st.markdown("#### Sentiment Volume Over Time")
                    sentiment_counts = time_df.groupby(['month', 'sentiment']).size().reset_index(name='count')
                    if not sentiment_counts.empty:
                        sentiment_stream_chart = px.area(
                            sentiment_counts, x='month', y='count', color='sentiment',
                            title="Volume of Reviews by Sentiment",
                            color_discrete_map={'Positive': 'green', 'Negative': 'red', 'Neutral': 'grey'},
                            category_orders={"sentiment": ["Positive", "Neutral", "Negative"]}
                        )
                        st.plotly_chart(sentiment_stream_chart, use_container_width=True)
    
        # ======================== KEYWORD ANALYSIS TAB ========================
        # =================== RATING BREAKDOWN TAB ========================
        with breakdown_tab:
            
            # This function is LOCAL and runs ONLY ONCE per product. It is extremely fast.
            @st.cache_data(show_spinner="Fetching rating breakdown...")
            def get_rating_distribution_data(_conn, asin):
                """
                Fetches pre-computed rating distribution data for a product.
                This query is instant and returns a single row.
                """
                return pd.read_sql("SELECT * FROM rating_distribution WHERE parent_asin = ?", _conn, params=(asin,))
    
            st.subheader("Overall Rating Breakdown")
            st.caption("This shows the exact count and proportion of all star ratings for this product.")
            
            distribution_df = get_rating_distribution_data(conn, selected_asin)
            
            if not distribution_df.empty:
                # Extract the first (and only) row of data
                dist_data = distribution_df.iloc[0]
                
                # Prepare data for the donut chart
                rating_values = {
                    "5 Stars": dist_data.get('5_star', 0),
                    "4 Stars": dist_data.get('4_star', 0),
                    "3 Stars": dist_data.get('3_star', 0),
                    "2 Stars": dist_data.get('2_star', 0),
                    "1 Star": dist_data.get('1_star', 0)
                }
                # Create a DataFrame from the dictionary
                donut_df = pd.DataFrame(list(rating_values.items()), columns=['Rating', 'Count'])
    
                col1, col2 = st.columns([1, 2])
    
                with col1:
                    st.markdown("#### Review Counts")
                    # Display exact counts using metrics for a clean look
                    st.metric("5 ⭐", f"{rating_values['5 Stars']:,} reviews")
                    st.metric("4 ⭐", f"{rating_values['4 Stars']:,} reviews")
                    st.metric("3 ⭐", f"{rating_values['3 Stars']:,} reviews")
                    st.metric("2 ⭐", f"{rating_values['2 Stars']:,} reviews")
                    st.metric("1 ⭐", f"{rating_values['1 Star']:,} reviews")
                
                with col2:
                    st.markdown("#### Rating Proportions")
                    # Create a donut chart with Plotly
                    fig = px.pie(
                        donut_df, 
                        values='Count', 
                        names='Rating', 
                        title='Proportion of Each Star Rating',
                        hole=0.4,
                        color_discrete_map={
                            '5 Stars': '#1a9850',
                            '4 Stars': '#91cf60',
                            '3 Stars': '#d9ef8b',
                            '2 Stars': '#fee08b',
                            '1 Star': '#d73027'
                        },
                        category_orders={'Rating': ["5 Stars", "4 Stars", "3 Stars", "2 Stars", "1 Star"]}
                    )
                    fig.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig, use_container_width=True)
        # ======================== WORD CLOUDS TAB ========================
        with wordcloud_tab:
            
            # This function is now LOCAL and runs ONLY ONCE per product, ignoring filters.
            @st.cache_data(show_spinner="Generating word clouds...")
            def generate_overall_word_frequency(_conn, asin, target_sentiment):
                """
                Generates word frequencies for ALL reviews of a product for a given sentiment.
                This is cached and runs only once, ensuring maximum performance.
                """
                query = f"SELECT text FROM reviews WHERE parent_asin = ? AND sentiment = ?"
                
                from spacy.lang.en.stop_words import STOP_WORDS
                word_counts = Counter()
                
                cursor = _conn.cursor()
                cursor.execute(query, (asin, target_sentiment))
    
                while True:
                    rows = cursor.fetchmany(1000)
                    if not rows:
                        break
                    for row in rows:
                        text = row[0]
                        if text:
                            words = re.findall(r'\b\w+\b', text.lower())
                            filtered_words = [word for word in words if word not in STOP_WORDS and len(word) > 2]
                            word_counts.update(filtered_words)
    
                if not word_counts:
                    return pd.DataFrame(columns=['word', 'freq'])
    
                freq_df = pd.DataFrame(word_counts.items(), columns=['word', 'freq']).sort_values(by='freq', ascending=False)
                return freq_df.head(100)
    
            st.subheader("Overall Word Clouds")
            st.caption("Showing the most common keywords from ALL positive and negative reviews for this product. These clouds do not react to sidebar filters.")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Key Themes in Positive Reviews")
                positive_freq_df = generate_overall_word_frequency(conn, selected_asin, 'Positive')
                if not positive_freq_df.empty:
                    fig = px.treemap(positive_freq_df, path=[px.Constant("Positive"), 'word'], values='freq', color_continuous_scale='Greens', color='freq')
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No positive reviews found for this product.")
            with col2:
                st.markdown("#### Key Themes in Negative Reviews")
                negative_freq_df = generate_overall_word_frequency(conn, selected_asin, 'Negative')
                if not negative_freq_df.empty:
                    fig = px.treemap(negative_freq_df, path=[px.Constant("Negative"), 'word'], values='freq', color_continuous_scale='Reds', color='freq')
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No negative reviews found for this product.")
    # --- MAIN SEARCH PAGE ---
    else:
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
