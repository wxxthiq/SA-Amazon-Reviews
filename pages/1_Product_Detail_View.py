# pages/1_Sentiment_Overview.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import altair as alt
from datetime import datetime
from utils.database_utils import (
    connect_to_db,
    get_product_details,
    get_reviews_for_product,
    get_product_date_range,
    get_single_review_details
)

# --- Page Configuration and State Initialization ---
st.set_page_config(layout="wide", page_title="Sentiment Overview")

# Initialize state keys if they don't exist
if 'selected_product' not in st.session_state:
    st.session_state.selected_product = None
if 'altair_selected_review_id' not in st.session_state:
    st.session_state.altair_selected_review_id = None

# --- Main App Logic ---
def main():
    st.title("📊 Sentiment Overview")

    # --- Constants & DB Connection ---
    DB_PATH = "amazon_reviews_top100.duckdb"
    PLACEHOLDER_IMAGE_URL = "https://via.placeholder.com/200"
    conn = connect_to_db(DB_PATH)

    # --- Check for Selected Product ---
    if st.session_state.selected_product is None:
        st.warning("Please select a product from the main search page first.")
        if st.button("⬅️ Back to Search"):
            st.switch_page("app.py")
        st.stop()
    selected_asin = st.session_state.selected_product

    # --- Load Product Data ---
    product_details_df = get_product_details(conn, selected_asin)
    if product_details_df.empty:
        st.error("Could not find details for the selected product.")
        if st.button("⬅️ Back to Search"):
            st.switch_page("app.py")
        st.stop()
    product_details = product_details_df.iloc[0]

    # --- Header Section ---
    if st.button("⬅️ Back to Search"):
        st.session_state.selected_product = None
        st.session_state.altair_selected_review_id = None
        st.switch_page("app.py")

    left_col, right_col = st.columns([1, 2])
    with left_col:
        image_urls_str = product_details.get('image_urls')
        image_urls = image_urls_str.split(',') if pd.notna(image_urls_str) and image_urls_str else []
        st.image(image_urls[0] if image_urls else PLACEHOLDER_IMAGE_URL, use_container_width=True)
        if image_urls:
            with st.popover("🖼️ View Image Gallery"):
                st.image(image_urls, use_container_width=True)
    with right_col:
        st.header(product_details['product_title'])
        st.caption(f"Category: {product_details['category']} | Store: {product_details['store']}")
        m_col1, m_col2 = st.columns(2)
        m_col1.metric("Average Rating", f"{product_details.get('average_rating', 0):.2f} ⭐")
        m_col2.metric("Total Reviews in DB", f"{int(product_details.get('review_count', 0)):,}")

    # --- Sidebar Filters ---
    st.sidebar.header("📊 Interactive Filters")
    min_date_db, max_date_db = get_product_date_range(conn, selected_asin)
    default_date_range = (min_date_db, max_date_db)
    default_ratings = [1, 2, 3, 4, 5]
    default_sentiments = ['Positive', 'Negative', 'Neutral']
    selected_date_range = st.sidebar.date_input("Filter by Date Range", value=default_date_range, min_value=min_date_db, max_value=max_date_db)
    selected_ratings = st.sidebar.multiselect("Filter by Star Rating", options=default_ratings, default=default_ratings)
    selected_sentiments = st.sidebar.multiselect("Filter by Sentiment", options=default_sentiments, default=default_sentiments)

    # --- Load Filtered Data ---
    chart_data = get_reviews_for_product(conn, selected_asin, selected_date_range, tuple(selected_ratings), tuple(selected_sentiments))

    st.markdown("---")

    if chart_data.empty:
        st.warning("No reviews match the selected filters.")
        st.stop()

    st.info(f"Displaying analysis for **{len(chart_data)}** reviews matching your criteria.")

    # --- Section 1: Distribution Charts ---
    # (Omitted for brevity, no changes here)
    st.markdown("### Key Distributions")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("#### Rating Distribution")
        rating_counts_df = chart_data['rating'].value_counts().sort_index().reset_index()
        bar_chart = alt.Chart(rating_counts_df).mark_bar().encode(x=alt.X('rating:O', title="Stars"), y=alt.Y('count:Q', title="Number of Reviews"), tooltip=['rating', 'count']).properties(height=300)
        st.altair_chart(bar_chart, use_container_width=True)
    with col2:
        st.markdown("#### Sentiment Distribution")
        sentiment_counts_df = chart_data['sentiment'].value_counts().reset_index()
        sentiment_chart = alt.Chart(sentiment_counts_df).mark_bar().encode(x=alt.X('sentiment:N', title="Sentiment", sort='-y'), y=alt.Y('count:Q', title="Number of Reviews"), color=alt.Color('sentiment:N', scale=alt.Scale(domain=['Positive', 'Neutral', 'Negative'], range=['#1a9850', '#cccccc', '#d73027']), legend=None), tooltip=['sentiment', 'count']).properties(height=300)
        st.altair_chart(sentiment_chart, use_container_width=True)
    with col3:
        st.markdown("#### Avg. Helpful Votes")
        helpful_df = chart_data.groupby('rating')['helpful_vote'].mean().reset_index()
        helpful_chart = alt.Chart(helpful_df).mark_bar(color='skyblue').encode(x=alt.X('rating:O', title='Star Rating'), y=alt.Y('helpful_vote:Q', title='Average Helpful Votes'), tooltip=['rating', 'helpful_vote']).properties(height=300)
        st.altair_chart(helpful_chart, use_container_width=True)

    # --- Section 2: Discrepancy Analysis (ROBUST ALTAIR IMPLEMENTATION) ---
    st.markdown("---")
    st.markdown("### Rating vs. Text Discrepancy")
    st.caption("Click a point on the chart to see the full review details on the right.")

    plot_col, review_col = st.columns([2, 1])

    with plot_col:
        chart_data['discrepancy'] = (chart_data['text_polarity'] - ((chart_data['rating'] - 3) / 2.0)).abs()
        
        # Define a point selection
        selection = alt.selection_point(fields=['review_id'], name='review_selector', empty=True)

        scatter_plot = alt.Chart(chart_data).mark_circle(size=60).encode(
            x=alt.X('rating:Q', title='Star Rating', scale=alt.Scale(zero=False)),
            y=alt.Y('text_polarity:Q', title='Text Sentiment Polarity'),
            color=alt.condition(selection, alt.value('orange'), 'discrepancy:Q', scale=alt.Scale(scheme='viridis'), legend=None),
            tooltip=['review_title', 'rating', 'text_polarity']
        ).add_params(
            selection
        ).properties(
            height=400
        )
        
        # Render the chart. on_select="rerun" will store the state in session_state
        st.altair_chart(scatter_plot, use_container_width=True, on_select="rerun", key="discrepancy_chart_selector")

        # After the chart is rendered, check the session state for the selection
        # This is the state object that Streamlit creates for the component
        selection_state = st.session_state.get("discrepancy_chart_selector")
        
        if selection_state and selection_state.get('review_id'):
            # The selection is a list, so we take the first item
            selected_id = selection_state['review_id'][0]
            # Update our custom state variable
            st.session_state.altair_selected_review_id = selected_id

    with review_col:
        # This display logic now only depends on our custom state variable
        if st.session_state.altair_selected_review_id:
            st.markdown("#### Selected Review Details")
            review_details = get_single_review_details(conn, st.session_state.altair_selected_review_id)
            
            if review_details is not None:
                st.subheader(review_details['review_title'])
                st.caption(f"Reviewed on: {review_details['date']}")
                st.markdown(f"> {review_details['text']}")
            else:
                st.
