# pages/1_Sentiment_Overview.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px # Keep for other charts
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

# Use a more specific key for the selection state
if 'altair_selected_review' not in st.session_state:
    st.session_state.altair_selected_review = None

# --- Main App Logic ---
def main():
    st.title("📊 Sentiment Overview")

    # --- Constants & DB Connection ---
    DB_PATH = "amazon_reviews_top100.duckdb"
    PLACEHOLDER_IMAGE_URL = "https://via.placeholder.com/200"
    conn = connect_to_db(DB_PATH)

    # --- Check for Selected Product ---
    if 'selected_product' not in st.session_state or st.session_state.selected_product is None:
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
        st.session_state.altair_selected_review = None # Clear state on exit
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
    st.markdown("### Key Distributions")
    # ... (unchanged)
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


    # --- Section 2: Discrepancy Analysis (NEW ALTAIR IMPLEMENTATION) ---
    st.markdown("---")
    st.markdown("### Rating vs. Text Discrepancy")
    st.caption("Click a point on the chart to see the full review details on the right.")

    plot_col, review_col = st.columns([2, 1])

    with plot_col:
        # Prepare data
        chart_data['discrepancy'] = (chart_data['text_polarity'] - ((chart_data['rating'] - 3) / 2.0)).abs()
        
        # Define the selection parameter
        selection = alt.selection_point(fields=['review_id'], empty=False)

        # Create the Altair chart
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
        
        # Use a key for the chart component to store its selection state
        event = st.altair_chart(scatter_plot, use_container_width=True, on_select="rerun", key="discrepancy_selector")
        
        # The selection state is stored in the key we provided
        if event.selection and event.selection['review_id']:
            st.session_state.altair_selected_review = event.selection['review_id'][0]

    with review_col:
        # Display logic now reads from the selection state populated by the chart
        if st.session_state.altair_selected_review:
            st.markdown("#### Selected Review Details")
            review_id = st.session_state.altair_selected_review
            review_details = get_single_review_details(conn, review_id)
            
            if review_details is not None:
                st.subheader(review_details['review_title'])
                st.caption(f"Reviewed on: {review_details['date']}")
                st.markdown(f"> {review_details['text']}")
            else:
                st.warning("Could not retrieve review details.")

            if st.button("Close Review", key="close_review_button_altair"):
                st.session_state.altair_selected_review = None
                st.rerun()
        else:
            st.info("Click a point on the plot to view details here.")

    # --- Section 3: Trend Analysis ---
    # ... (unchanged)
    st.markdown("---")
    st.markdown("### Trends Over Time")
    time_df = chart_data.copy()
    time_df['date'] = pd.to_datetime(time_df['date'])
    time_df['month'] = time_df['date'].dt.to_period('M').dt.start_time
    t_col1, t_col2 = st.columns(2)
    with t_col1:
        st.markdown("#### Volume of Reviews")
        review_counts_over_time = time_df.groupby('month').size().reset_index(name='count')
        if not review_counts_over_time.empty:
            review_stream_chart = px.area(review_counts_over_time, x='month', y='count', title="Total Reviews Published Per Month")
            st.plotly_chart(review_stream_chart, use_container_width=True)
    with t_col2:
        st.markdown("#### Volume of Sentiments")
        sentiment_counts_over_time = time_df.groupby(['month', 'sentiment']).size().reset_index(name='count')
        if not sentiment_counts_over_time.empty:
            sentiment_stream_chart = px.area(sentiment_counts_over_time, x='month', y='count', color='sentiment', title="Sentiment Breakdown Per Month", color_discrete_map={'Positive': '#1a9850', 'Neutral': '#cccccc', 'Negative': '#d73027'}, category_orders={"sentiment": ["Positive", "Neutral", "Negative"]})
            st.plotly_chart(sentiment_stream_chart, use_container_width=True)


if __name__ == "__main__":
    main()
