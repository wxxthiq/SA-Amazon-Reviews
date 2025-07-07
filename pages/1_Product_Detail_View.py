# pages/1_Product_Detail_View.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import altair as alt
from datetime import datetime
from utils.database_utils import connect_to_db, get_product_details, get_reviews_for_product, get_product_date_range

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Product Analysis")
DB_PATH = "amazon_reviews_top100.duckdb"
PLACEHOLDER_IMAGE_URL = "https://via.placeholder.com/200"

# --- Connect to DB ---
conn = connect_to_db(DB_PATH)

# --- Check for Selected Product ---
if 'selected_product' not in st.session_state or st.session_state.selected_product is None:
    st.warning("Please select a product from the main search page first.")
    st.page_link("app.py", label="Back to Search", icon="⬅️")
    st.stop()

selected_asin = st.session_state.selected_product

# --- Load Product Data ---
product_details_df = get_product_details(conn, selected_asin)
if product_details_df.empty:
    st.error("Could not find details for the selected product.")
    st.page_link("app.py", label="Back to Search", icon="⬅️")
    st.stop()
product_details = product_details_df.iloc[0]

# --- RENDER PAGE ---

# --- Header and Sidebar ---
if st.page_link("app.py", label="Back to Search", icon="⬅️"):
    # Clear product-specific state when going back
    st.session_state.selected_product = None

# Product Info Header
left_col, right_col = st.columns([1, 2])
with left_col:
    image_urls_str = product_details.get('image_urls')
    image_urls = image_urls_str.split(',') if pd.notna(image_urls_str) and image_urls_str else []
    st.image(image_urls[0] if image_urls else PLACEHOLDER_IMAGE_URL, use_container_width=True)

with right_col:
    st.header(product_details['product_title'])
    st.caption(f"Category: {product_details['category']} | Store: {product_details['store']}")
    st.metric("Average Rating", f"{product_details.get('average_rating', 0):.2f} ⭐")
    st.metric("Total Reviews in DB", f"{int(product_details.get('review_count', 0)):,}")

# --- Interactive Sidebar Filters ---
st.sidebar.header("📊 Interactive Filters")
min_date_db, max_date_db = get_product_date_range(conn, selected_asin)

# Set defaults for filters
default_date_range = (min_date_db, max_date_db)
default_ratings = [1, 2, 3, 4, 5]
default_sentiments = ['Positive', 'Negative', 'Neutral']

# Create filter widgets
selected_date_range = st.sidebar.date_input(
    "Filter by Date Range",
    value=default_date_range,
    min_value=min_date_db,
    max_value=max_date_db,
    key='date_filter'
)
selected_ratings = st.sidebar.multiselect(
    "Filter by Star Rating",
    options=default_ratings,
    default=default_ratings,
    key='rating_filter'
)
selected_sentiments = st.sidebar.multiselect(
    "Filter by Sentiment",
    options=default_sentiments,
    default=default_sentiments,
    key='sentiment_filter'
)

# --- Load Filtered Data ---
# This is the main data pull for all charts on this page
chart_data = get_reviews_for_product(
    conn,
    selected_asin,
    selected_date_range,
    tuple(selected_ratings),  # Use tuple for caching
    tuple(selected_sentiments) # Use tuple for caching
)

st.markdown("---")
st.subheader("Sentiment Analysis Overview")

if chart_data.empty:
    st.warning("No reviews match the selected filters.")
else:
    st.info(f"Displaying analysis for **{len(chart_data)}** reviews matching your criteria.")
    
    # --- RENDER CHARTS ---
    col1, col2 = st.columns(2)
    
    with col1:
        # Rating Distribution Chart
        st.markdown("#### Rating Distribution")
        rating_counts_df = chart_data['rating'].value_counts().sort_index().reset_index()
        rating_counts_df.columns = ['Rating', 'Count']
        bar_chart = alt.Chart(rating_counts_df).mark_bar().encode(
            x=alt.X('Rating:O', title="Stars"),
            y=alt.Y('Count:Q', title="Number of Reviews"),
            tooltip=['Rating', 'Count']
        ).properties(title="Filtered Rating Distribution")
        st.altair_chart(bar_chart, use_container_width=True)

    with col2:
        # Sentiment Distribution Chart
        st.markdown("#### Sentiment Distribution")
        sentiment_counts_df = chart_data['sentiment'].value_counts().reset_index()
        sentiment_counts_df.columns = ['Sentiment', 'Count']
        sentiment_chart = alt.Chart(sentiment_counts_df).mark_bar().encode(
            x=alt.X('Sentiment:N', title="Sentiment", sort='-y'),
            y=alt.Y('Count:Q', title="Number of Reviews"),
            color=alt.Color('Sentiment:N',
                            scale=alt.Scale(domain=['Positive', 'Neutral', 'Negative'], range=['#1a9850', '#cccccc', '#d73027']),
                            legend=None),
            tooltip=['Sentiment', 'Count']
        ).properties(title="Filtered Sentiment Distribution")
        st.altair_chart(sentiment_chart, use_container_width=True)

    # Helpfulness Analysis (NEW FEATURE)
    st.markdown("---")
    st.markdown("### 👍 Helpfulness Analysis")
    col3, col4 = st.columns(2)

    with col3:
        # Average helpful votes per rating
        st.markdown("#### Avg. Helpful Votes per Rating")
        helpful_df = chart_data.groupby('rating')['helpful_vote'].mean().reset_index()
        helpful_chart = alt.Chart(helpful_df).mark_bar(color='skyblue').encode(
            x=alt.X('rating:O', title='Star Rating'),
            y=alt.Y('helpful_vote:Q', title='Average Helpful Votes'),
            tooltip=['rating', 'helpful_vote']
        ).properties(title="Do higher or lower ratings get more helpful votes?")
        st.altair_chart(helpful_chart, use_container_width=True)

    with col4:
        # Verified vs Unverified reviews helpfulness
        st.markdown("#### Verified vs. Unverified Reviews")
        verified_df = chart_data.groupby('verified_purchase')['helpful_vote'].agg(['mean', 'count']).reset_index()
        verified_df['verified_purchase'] = verified_df['verified_purchase'].map({True: 'Verified', False: 'Not Verified'})
        
        # Ensure we have data to plot
        if not verified_df.empty:
            verified_chart = alt.Chart(verified_df).mark_bar().encode(
                x=alt.X('verified_purchase:N', title='Purchase Status'),
                y=alt.Y('mean:Q', title='Average Helpful Votes'),
                tooltip=['verified_purchase', 'mean', 'count']
            ).properties(
                title='Are verified reviews more helpful?'
            )
            st.altair_chart(verified_chart, use_container_width=True)
        else:
            st.info("No data available for verified vs. unverified comparison with current filters.")
