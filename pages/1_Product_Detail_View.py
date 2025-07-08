# pages/1_Sentiment_Overview.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import altair as alt
from datetime import datetime
from streamlit_plotly_events import plotly_events
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from collections import Counter
import re

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
    conn = connect_to_db(DB_PATH)

    # --- Product and Data Loading ---
    if 'selected_product' not in st.session_state or st.session_state.selected_product is None:
        st.warning("Please select a product from the main search page first.")
        st.stop()
    selected_asin = st.session_state.selected_product
    product_details_df = get_product_details(conn, selected_asin)
    if product_details_df.empty:
        st.error("Could not find details for the selected product.")
        st.stop()
    product_details = product_details_df.iloc[0]

    # --- Sidebar Filters ---
    st.sidebar.header("📊 Interactive Filters")
    def reset_selection():
        st.session_state.selected_review_id = None
    min_date_db, max_date_db = get_product_date_range(conn, selected_asin)
    default_date_range = (min_date_db, max_date_db)
    default_ratings = [1, 2, 3, 4, 5]
    default_sentiments = ['Positive', 'Negative', 'Neutral']
    
    selected_date_range = st.sidebar.date_input("Filter by Date Range", value=default_date_range, key='date_filter', on_change=reset_selection)
    selected_ratings = st.sidebar.multiselect("Filter by Star Rating", options=default_ratings, default=default_ratings, key='rating_filter', on_change=reset_selection)
    selected_sentiments = st.sidebar.multiselect("Filter by Sentiment", options=default_sentiments, default=default_sentiments, key='sentiment_filter', on_change=reset_selection)

    def reset_all_filters():
        st.session_state.date_filter = default_date_range
        st.session_state.rating_filter = default_ratings
        st.session_state.sentiment_filter = default_sentiments
        st.session_state.selected_review_id = None
    st.sidebar.button("Reset All Filters", on_click=reset_all_filters, use_container_width=True)

    # --- Load Filtered Data ---
    chart_data = get_reviews_for_product(conn, selected_asin, selected_date_range, tuple(selected_ratings), tuple(selected_sentiments))

    # --- Header Section ---
    # ... (code is unchanged, omitted for brevity)
    if st.button("⬅️ Back to Search"):
        st.session_state.selected_product = None
        st.session_state.selected_review_id = None
        st.switch_page("app.py")
    left_col, right_col = st.columns([1, 2])
    with left_col:
        st.image(product_details.get('image_urls', '').split(',')[0] if product_details.get('image_urls') else "https://via.placeholder.com/200", use_container_width=True)
    with right_col:
        st.header(product_details['product_title'])
        st.caption(f"Category: {product_details['category']} | Store: {product_details['store']}")
        m_col1, m_col2 = st.columns(2)
        m_col1.metric("Average Rating", f"{product_details.get('average_rating', 0):.2f} ⭐")
        m_col2.metric("Filtered Reviews", f"{len(chart_data):,}")
        st.markdown("---")
        dist_col1, dist_col2 = st.columns(2)
        with dist_col1:
            st.markdown("**Rating Distribution**")
            if not chart_data.empty:
                rating_counts = chart_data['rating'].value_counts().reindex(range(1, 6), fill_value=0)
                total_ratings = len(chart_data)
                for rating in range(5, 0, -1):
                    count = rating_counts.get(rating, 0)
                    percentage = (count / total_ratings * 100) if total_ratings > 0 else 0
                    st.text(f"{rating} ⭐: {percentage:.1f}% ({count})")
                    st.progress(int(percentage))
        with dist_col2:
            st.markdown("**Sentiment Distribution**")
            if not chart_data.empty:
                sentiment_counts = chart_data['sentiment'].value_counts()
                total_sentiments = len(chart_data)
                sentiment_colors = {"Positive": "green", "Neutral": "grey", "Negative": "red"}
                for sentiment in ['Positive', 'Neutral', 'Negative']:
                    count = sentiment_counts.get(sentiment, 0)
                    percentage = (count / total_sentiments * 100) if total_sentiments > 0 else 0
                    st.markdown(f":{sentiment_colors.get(sentiment, 'default')}[{sentiment}]: {percentage:.1f}% ({count})")
                    st.progress(int(percentage))
    
    st.markdown("---")
    if chart_data.empty:
        st.warning("No reviews match the selected filters.")
        st.stop()
    st.info(f"Displaying analysis for **{len(chart_data)}** reviews matching your criteria.")

    # --- Discrepancy & Trend Analysis Sections (Unchanged) ---
    # ... (code omitted for brevity)
    st.markdown("### Rating vs. Text Discrepancy")
    plot_col, review_col = st.columns([2, 1])
    with plot_col:
        # ... (code omitted for brevity)
        chart_data['discrepancy'] = (chart_data['text_polarity'] - ((chart_data['rating'] - 3) / 2.0)).abs()
        fig = px.scatter(chart_data, x="rating_jittered", y="text_polarity_jittered", color="discrepancy", color_continuous_scale=px.colors.sequential.Viridis, hover_name='review_title')
        fig.update_layout(clickmode='event+select')
        fig.update_traces(marker_size=10)
        selected_points = plotly_events(fig, click_event=True, key="plotly_event_selector")
        if selected_points and 'pointIndex' in selected_points[0]:
            point_index = selected_points[0]['pointIndex']
            if point_index < len(chart_data):
                clicked_id = chart_data.iloc[point_index]['review_id']
                if st.session_state.selected_review_id != clicked_id:
                    st.session_state.selected_review_id = clicked_id
                    st.rerun()
    with review_col:
        # ... (code omitted for brevity)
        if st.session_state.selected_review_id:
            if st.session_state.selected_review_id in chart_data['review_id'].values:
                st.markdown("#### Selected Review Details")
                review_details = get_single_review_details(conn, st.session_state.selected_review_id)
                if review_details is not None:
                    st.subheader(review_details['review_title'])
                    st.caption(f"Reviewed on: {review_details['date']}")
                    st.markdown(f"> {review_details['text']}")
                if st.button("Close Review", key="close_review_button"):
                    st.session_state.selected_review_id = None
                    st.rerun()
    st.markdown("---")
    st.markdown("### Trends Over Time")
    time_granularity = st.radio("Select time period:", ("Monthly", "Weekly", "Daily"), index=0, horizontal=True, label_visibility="collapsed")
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
        st.markdown("#### Rating Distribution Over Time")
        rating_counts_over_time = time_df.groupby(['period', 'rating']).size().reset_index(name='count')
        if not rating_counts_over_time.empty:
            rating_stream_chart = px.area(rating_counts_over_time, x='period', y='count', color='rating', title=f"Volume of Reviews by Star Rating", color_discrete_map={5: '#1a9850', 4: '#91cf60', 3: '#d9ef8b', 2: '#fee08b', 1: '#d73027'}, category_orders={"rating": [5, 4, 3, 2, 1]})
            st.plotly_chart(rating_stream_chart, use_container_width=True)
    with t_col2:
        st.markdown("#### Sentiment Volume Over Time")
        sentiment_counts_over_time = time_df.groupby(['period', 'sentiment']).size().reset_index(name='count')
        if not sentiment_counts_over_time.empty:
            sentiment_stream_chart = px.area(sentiment_counts_over_time, x='period', y='count', color='sentiment', title=f"Sentiment Breakdown Per {time_granularity.replace('ly', '')}", color_discrete_map={'Positive': '#1a9850', 'Neutral': '#cccccc', 'Negative': '#d73027'}, category_orders={"sentiment": ["Positive", "Neutral", "Negative"]})
            st.plotly_chart(sentiment_stream_chart, use_container_width=True)


    # --- NEW KEYWORD ANALYSIS SECTION ---
    st.markdown("---")
    st.markdown("### 🔑 Keyword Analysis")
    st.caption("Explore the most common themes in positive and negative reviews.")

    # Helper function to get top keywords
    @st.cache_data
    def get_top_keywords(text_series, n=10):
        # Combine all text, clean it, and count words
        all_text = ' '.join(text_series.astype(str))
        words = re.findall(r'\b\w+\b', all_text.lower())
        # Use a more extensive list of stopwords
        custom_stopwords = set(STOPWORDS) | {'product', 'review', 'item', 'im', 'ive', 'id', 'get', 'it', 'the', 'and', 'but'}
        filtered_words = [word for word in words if word not in custom_stopwords and len(word) > 2]
        return [word for word, count in Counter(filtered_words).most_common(n)]

    # Get and display top keywords
    positive_text = chart_data[chart_data["sentiment"] == "Positive"]["text"]
    negative_text = chart_data[chart_data["sentiment"] == "Negative"]["text"]

    top_pos_keywords = get_top_keywords(positive_text)
    top_neg_keywords = get_top_keywords(negative_text)

    kw_col1, kw_col2 = st.columns(2)
    with kw_col1:
        st.markdown("#### Top Positive Keywords")
        st.info(" ".join(f"`{word}`" for word in top_pos_keywords))
    with kw_col2:
        st.markdown("#### Top Negative Keywords")
        st.error(" ".join(f"`{word}`" for word in top_neg_keywords))

    # --- Interactive Keyword Explorer ---
    st.markdown("---")
    all_top_keywords = sorted(list(set(top_pos_keywords + top_neg_keywords)))
    
    selected_keyword = st.selectbox(
        "Select a keyword to analyze:",
        options=["--- Select a Keyword ---"] + all_top_keywords
    )

    if selected_keyword != "--- Select a Keyword ---":
        # Filter reviews containing the selected keyword
        keyword_df = chart_data[chart_data['text'].str.contains(r'\b' + selected_keyword + r'\b', case=False, na=False)]
        
        st.markdown(f"#### Analysis for keyword: `{selected_keyword}` ({len(keyword_df)} mentions)")

        # Display rating distribution for the keyword
        dist_chart_col, _ = st.columns([2,1])
        with dist_chart_col:
            st.markdown("**Rating Distribution for this Keyword**")
            rating_dist = keyword_df['rating'].value_counts().sort_index()
            st.bar_chart(rating_dist)

        # Display 5 reviews with sorting
        st.markdown("**Example Reviews**")
        sort_reviews_by = st.selectbox("Sort examples by:", ["Most Helpful", "Newest"], key="keyword_review_sort")
        
        if sort_reviews_by == "Most Helpful":
            sorted_keyword_df = keyword_df.sort_values(by="helpful_vote", ascending=False)
        else: # Newest
            sorted_keyword_df = keyword_df.sort_values(by="date", ascending=False)

        for _, review in sorted_keyword_df.head(5).iterrows():
            with st.container(border=True):
                st.caption(f"**Rating: {review['rating']} ⭐ | Helpful Votes: {review['helpful_vote']}**")
                st.markdown(f"> {review['text']}")
    
    # --- Navigation to Review Explorer ---
    st.markdown("---")
    st.subheader("📝 Browse Individual Reviews")
    st.markdown("Click the button below to browse, sort, and filter all reviews for this product.")
    if st.button("Explore All Reviews"):
        st.switch_page("pages/2_Review_Explorer.py")


if __name__ == "__main__":
    main()
