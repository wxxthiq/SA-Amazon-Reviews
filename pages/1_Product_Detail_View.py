# pages/1_Sentiment_Overview.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import altair as alt
from datetime import datetime
from streamlit_plotly_events import plotly_events

from utils.database_utils import (
    connect_to_db,
    get_product_details,
    get_reviews_for_product,
    get_product_date_range,
    get_single_review_details
)

# --- Page Configuration and State Initialization ---
st.set_page_config(layout="wide", page_title="Sentiment Overview")

if 'selected_review_id' not in st.session_state:
    st.session_state.selected_review_id = None

# --- Main App Logic ---
def main():
    st.title("📊 Sentiment Overview")

    # --- Constants & DB Connection ---
    DB_PATH = "amazon_reviews_top100.duckdb"
    PLACEHOLDER_IMAGE_URL = "https://via.placeholder.com/200"
    conn = connect_to_db(DB_PATH)

    # --- Product and Data Loading ---
    if 'selected_product' not in st.session_state or st.session_state.selected_product is None:
        st.warning("Please select a product from the main search page first.")
        if st.button("⬅️ Back to Search"):
            st.switch_page("app.py")
        st.stop()
    selected_asin = st.session_state.selected_product

    product_details_df = get_product_details(conn, selected_asin)
    if product_details_df.empty:
        st.error("Could not find details for the selected product.")
        if st.button("⬅️ Back to Search"):
            st.switch_page("app.py")
        st.stop()
    product_details = product_details_df.iloc[0]

    # --- Header Section (Unchanged) ---
    if st.button("⬅️ Back to Search"):
        st.session_state.selected_product = None
        st.session_state.selected_review_id = None
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

    # --- Sidebar Filters (WITH STATE RESET) ---
    st.sidebar.header("📊 Interactive Filters")
    
    # ** KEY CHANGE: Define a callback function to reset the selection **
    def reset_selection():
        st.session_state.selected_review_id = None

    min_date_db, max_date_db = get_product_date_range(conn, selected_asin)
    default_date_range = (min_date_db, max_date_db)
    default_ratings = [1, 2, 3, 4, 5]
    default_sentiments = ['Positive', 'Negative', 'Neutral']

    # ** KEY CHANGE: Add the on_change callback to all filter widgets **
    selected_date_range = st.sidebar.date_input("Filter by Date Range", value=default_date_range, min_value=min_date_db, max_value=max_date_db, on_change=reset_selection)
    selected_ratings = st.sidebar.multiselect("Filter by Star Rating", options=default_ratings, default=default_ratings, on_change=reset_selection)
    selected_sentiments = st.sidebar.multiselect("Filter by Sentiment", options=default_sentiments, default=default_sentiments, on_change=reset_selection)

    # --- Load Filtered Data (Now includes stable jitter) ---
    chart_data = get_reviews_for_product(conn, selected_asin, selected_date_range, tuple(selected_ratings), tuple(selected_sentiments))

    st.markdown("---")

    if chart_data.empty:
        st.warning("No reviews match the selected filters.")
        st.stop()

    st.info(f"Displaying analysis for **{len(chart_data)}** reviews matching your criteria.")
    
    # --- Distribution Charts (Unchanged) ---
    st.markdown("### Key Distributions")
    # ... (Code omitted for brevity)
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

    # --- Section 2: Discrepancy Analysis (Now with stable plot) ---
    st.markdown("---")
    st.markdown("### Rating vs. Text Discrepancy")
    st.caption("Click a point on the chart to see the full review details on the right.")

    plot_col, review_col = st.columns([2, 1])

    with plot_col:
        # ** KEY CHANGE: Jitter is now pre-calculated, so we just calculate discrepancy **
        chart_data['discrepancy'] = (chart_data['text_polarity'] - ((chart_data['rating'] - 3) / 2.0)).abs()
        
        fig = px.scatter(
            chart_data,
            x="rating_jittered", y="text_polarity_jittered",
            color="discrepancy", color_continuous_scale=px.colors.sequential.Viridis,
            hover_name='review_title'
        )
        fig.update_layout(clickmode='event+select')
        fig.update_traces(marker_size=10)

        selected_points = plotly_events(fig, click_event=True, key="plotly_event_selector")

        if selected_points and 'pointIndex' in selected_points[0]:
            point_index = selected_points[0]['pointIndex']
            # Safely check if the index is still valid for the current chart_data
            if point_index < len(chart_data):
                clicked_id = chart_data.iloc[point_index]['review_id']
                if st.session_state.selected_review_id != clicked_id:
                    st.session_state.selected_review_id = clicked_id
                    st.rerun()

    with review_col:
        if st.session_state.selected_review_id:
            # ** KEY CHANGE: Add a check to ensure the selected ID is still in the filtered data **
            if st.session_state.selected_review_id in chart_data['review_id'].values:
                st.markdown("#### Selected Review Details")
                review_details = get_single_review_details(conn, st.session_state.selected_review_id)
                
                if review_details is not None:
                    st.subheader(review_details['review_title'])
                    st.caption(f"Reviewed on: {review_details['date']}")
                    st.markdown(f"> {review_details['text']}")
                else:
                    st.warning("Could not retrieve review details.")

                if st.button("Close Review", key="close_review_button"):
                    st.session_state.selected_review_id = None
                    st.rerun()
            else:
                # If the selected review is no longer in the filtered data, show a message.
                st.info("The previously selected review is not visible with the current filters. Please select a new point.")
        else:
            st.info("Click a point on the plot to view details here.")
            
    # --- Section 3: Trend Analysis (ENHANCED) ---
    st.markdown("---")
    st.markdown("### Trends Over Time")

    time_granularity = st.radio(
        "Select time period:",
        ("Monthly", "Weekly", "Daily"),
        index=0,
        horizontal=True,
        label_visibility="collapsed"
    )

    time_df = chart_data.copy()
    time_df['date'] = pd.to_datetime(time_df['date'])

    if time_granularity == 'Monthly':
        time_df['period'] = time_df['date'].dt.to_period('M').dt.start_time
    elif time_granularity == 'Weekly':
        time_df['period'] = time_df['date'].dt.to_period('W').dt.start_time
    else: # Daily
        time_df['period'] = time_df['date'].dt.date

    t_col1, t_col2 = st.columns(2)

    with t_col1:
        # ** KEY CHANGE: Switched to Rating Distribution **
        st.markdown("#### Rating Distribution Over Time")
        rating_counts_over_time = time_df.groupby(['period', 'rating']).size().reset_index(name='count')
        if not rating_counts_over_time.empty:
            rating_stream_chart = px.area(
                rating_counts_over_time,
                x='period',
                y='count',
                color='rating',
                title=f"Volume of Reviews by Star Rating",
                color_discrete_map={
                    5: '#1a9850', 4: '#91cf60', 3: '#d9ef8b',
                    2: '#fee08b', 1: '#d73027'
                },
                category_orders={"rating": [5, 4, 3, 2, 1]}
            )
            st.plotly_chart(rating_stream_chart, use_container_width=True)

    with t_col2:
        st.markdown("#### Sentiment Volume Over Time")
        sentiment_counts_over_time = time_df.groupby(['period', 'sentiment']).size().reset_index(name='count')
        if not sentiment_counts_over_time.empty:
            sentiment_stream_chart = px.area(
                sentiment_counts_over_time, x='period', y='count', color='sentiment',
                title=f"Sentiment Breakdown Per {time_granularity.replace('ly', '')}",
                color_discrete_map={'Positive': '#1a9850', 'Neutral': '#cccccc', 'Negative': '#d73027'},
                category_orders={"sentiment": ["Positive", "Neutral", "Negative"]}
            )
            st.plotly_chart(sentiment_stream_chart, use_container_width=True)
            
    # --- Section 4: Navigation to Review Explorer ---
    st.markdown("---")
    st.subheader("📝 Browse Individual Reviews")
    st.markdown("Click the button below to browse, sort, and filter all reviews for this product.")

    if st.button("Explore All Reviews"):
        st.switch_page("pages/2_Review_Explorer.py")
        
if __name__ == "__main__":
    main()
