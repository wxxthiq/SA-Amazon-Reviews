# pages/3_Keyword_Analysis.py
import streamlit as st
import pandas as pd
import re
from collections import Counter
from wordcloud import STOPWORDS
from utils.database_utils import (
    connect_to_db,
    get_product_details,
    get_reviews_for_product
)

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Keyword Analysis")
DB_PATH = "amazon_reviews_top100.duckdb"
conn = connect_to_db(DB_PATH)

# --- Main App Logic ---
def main():
    st.title("🔑 Detailed Keyword Analysis")

    if st.button("⬅️ Back to Sentiment Overview"):
        st.switch_page("pages/1_Sentiment_Overview.py")

    # --- Check for Selected Product ---
    if 'selected_product' not in st.session_state or st.session_state.selected_product is None:
        st.warning("Please select a product from the main search page first.")
        st.stop()
    selected_asin = st.session_state.selected_product

    # --- Load Product Data ---
    product_details = get_product_details(conn, selected_asin).iloc[0]
    st.header(product_details['product_title'])
    st.caption("Select a keyword to see its rating distribution and read example reviews.")

    # --- Use existing filters from session state ---
    # This ensures that the analysis is consistent with the overview page
    date_filter = st.session_state.get('date_filter')
    rating_filter = st.session_state.get('rating_filter')
    sentiment_filter = st.session_state.get('sentiment_filter')

    # Load data based on the filters set on the overview page
    chart_data = get_reviews_for_product(conn, selected_asin, date_filter, tuple(rating_filter), tuple(sentiment_filter))

    if chart_data.empty:
        st.warning("No review data available for the selected filters. Please adjust them on the overview page.")
        st.stop()

    # --- Keyword Extraction and Selection ---
    @st.cache_data
    def get_top_keywords(text_series, n=15):
        all_text = ' '.join(text_series.astype(str))
        words = re.findall(r'\b\w+\b', all_text.lower())
        custom_stopwords = set(STOPWORDS) | {'product', 'review', 'item', 'im', 'ive', 'id', 'get', 'it', 'the', 'and', 'but'}
        filtered_words = [word for word in words if word not in custom_stopwords and len(word) > 2]
        return [word for word, count in Counter(filtered_words).most_common(n)]

    positive_text = chart_data[chart_data["sentiment"] == "Positive"]["text"]
    negative_text = chart_data[chart_data["sentiment"] == "Negative"]["text"]
    top_pos_keywords = get_top_keywords(positive_text)
    top_neg_keywords = get_top_keywords(negative_text)
    all_top_keywords = sorted(list(set(top_pos_keywords + top_neg_keywords)))

    selected_keyword = st.selectbox(
        "Select a keyword to analyze:",
        options=["--- Select a Keyword ---"] + all_top_keywords
    )

    # --- Display Keyword Analysis ---
    if selected_keyword != "--- Select a Keyword ---":
        keyword_df = chart_data[chart_data['text'].str.contains(r'\b' + selected_keyword + r'\b', case=False, na=False)]
        
        st.markdown(f"---")
        st.markdown(f"#### Analysis for keyword: `{selected_keyword}` ({len(keyword_df)} mentions)")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Rating Distribution for this Keyword**")
            rating_dist = keyword_df['rating'].value_counts().reindex(range(1, 6), fill_value=0).sort_index()
            st.bar_chart(rating_dist)
        with col2:
            st.markdown("**Sentiment Distribution for this Keyword**")
            sentiment_dist = keyword_df['sentiment'].value_counts().reindex(['Positive', 'Neutral', 'Negative'], fill_value=0)
            st.bar_chart(sentiment_dist)

        st.markdown("**Example Reviews**")
        sort_reviews_by = st.selectbox("Sort examples by:", ["Most Helpful", "Newest"], key="keyword_review_sort")
        
        if sort_reviews_by == "Most Helpful":
            sorted_keyword_df = keyword_df.sort_values(by="helpful_vote", ascending=False)
        else:
            sorted_keyword_df = keyword_df.sort_values(by="date", ascending=False)

        for _, review in sorted_keyword_df.head(5).iterrows():
            with st.container(border=True):
                st.caption(f"**Rating: {review['rating']} ⭐ | Helpful Votes: {review['helpful_vote']} | Sentiment: {review['sentiment']}**")
                st.markdown(f"> {review['text']}")

if __name__ == "__main__":
    main()
