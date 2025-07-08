# pages/3_Keyword_Analysis.py
import streamlit as st
import pandas as pd
import re
from collections import Counter
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from utils.database_utils import (
    connect_to_db,
    get_product_details,
    get_reviews_for_product,
    get_product_date_range
)

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Keyword Analysis")
DB_PATH = "amazon_reviews_top100.duckdb"
conn = connect_to_db(DB_PATH)
REVIEWS_PER_PAGE = 5

# --- Main App Logic ---
def main():
    st.title("🔑 Detailed Keyword Analysis")

    if st.button("⬅️ Back to Sentiment Overview"):
        st.switch_page("pages/1_Sentiment_Overview.py")

    # (Code for product loading and sidebar filters is unchanged)
    # ...
    if 'selected_product' not in st.session_state or st.session_state.selected_product is None:
        st.warning("Please select a product from the main search page first.")
        st.stop()
    selected_asin = st.session_state.selected_product
    product_details = get_product_details(conn, selected_asin).iloc[0]
    st.header(product_details['product_title'])
    st.caption("Use the sidebar to filter the reviews, then select a keyword to analyze.")
    st.sidebar.header("🔬 Keyword Analysis Filters")
    min_date_db, max_date_db = get_product_date_range(conn, selected_asin)
    default_date_range = (min_date_db, max_date_db)
    default_ratings = [1, 2, 3, 4, 5]
    default_sentiments = ['Positive', 'Negative', 'Neutral']
    default_verified = "All"
    if 'keyword_date_filter' not in st.session_state:
        st.session_state.keyword_date_filter = default_date_range
    if 'keyword_rating_filter' not in st.session_state:
        st.session_state.keyword_rating_filter = default_ratings
    if 'keyword_sentiment_filter' not in st.session_state:
        st.session_state.keyword_sentiment_filter = default_sentiments
    if 'keyword_verified_filter' not in st.session_state:
        st.session_state.keyword_verified_filter = default_verified
    def reset_keyword_page():
        st.session_state.keyword_review_page = 0
    st.sidebar.date_input("Filter by Date Range", key='keyword_date_filter', on_change=reset_keyword_page)
    st.sidebar.multiselect("Filter by Star Rating", options=default_ratings, key='keyword_rating_filter', on_change=reset_keyword_page)
    st.sidebar.multiselect("Filter by Sentiment", options=default_sentiments, key='keyword_sentiment_filter', on_change=reset_keyword_page)
    st.sidebar.radio("Filter by Purchase Status", ["All", "Verified Only", "Not Verified"], key='keyword_verified_filter', on_change=reset_keyword_page)
    chart_data = get_reviews_for_product(conn, selected_asin, st.session_state.keyword_date_filter, tuple(st.session_state.keyword_rating_filter), tuple(st.session_state.keyword_sentiment_filter), st.session_state.keyword_verified_filter)
    st.markdown("---")
    if chart_data.empty:
        st.warning("No review data available for the selected filters.")
        st.stop()
    st.info(f"Analyzing keywords from **{len(chart_data)}** reviews matching your criteria.")

    # --- WORD CLOUD SUMMARY (WITH SLIDER) ---
    st.markdown("### ☁️ Keyword Summary")

    # ** NEW: Slider to control the number of words **
    max_words = st.slider(
        "Select the max number of words to display in the clouds:",
        min_value=5, max_value=50, value=15,
        key='keyword_max_words'
    )
    
    wc_col1, wc_col2 = st.columns(2)
    with wc_col1:
        st.markdown("#### Positive Keywords")
        pos_text = " ".join(review for review in chart_data[chart_data["sentiment"]=="Positive"]["text"])
        if pos_text:
            wordcloud_pos = WordCloud(
                stopwords=STOPWORDS, background_color="white", width=800, height=400, colormap='Greens',
                max_words=max_words # Use the slider value here
            ).generate(pos_text)
            fig, ax = plt.subplots()
            ax.imshow(wordcloud_pos, interpolation='bilinear')
            ax.axis("off")
            st.pyplot(fig)
    with wc_col2:
        st.markdown("#### Negative Keywords")
        neg_text = " ".join(review for review in chart_data[chart_data["sentiment"]=="Negative"]["text"])
        if neg_text:
            wordcloud_neg = WordCloud(
                stopwords=STOPWORDS, background_color="white", width=800, height=400, colormap='Reds',
                max_words=max_words # Use the slider value here
            ).generate(neg_text)
            fig, ax = plt.subplots()
            ax.imshow(wordcloud_neg, interpolation='bilinear')
            ax.axis("off")
            st.pyplot(fig)

    # --- INTERACTIVE KEYWORD EXPLORER ---
    # ... (rest of the file is unchanged, omitted for brevity)
    st.markdown("---")
    st.markdown("### 🔬 Interactive Keyword Explorer")
    @st.cache_data
    def get_top_keywords_by_mention(text_series, n=25):
        mention_counter = Counter()
        custom_stopwords = set(STOPWORDS) | {'product', 'review', 'item', 'im', 'ive', 'id', 'get', 'it', 'the', 'and', 'but', 'use', 'one'}
        for text in text_series:
            words = re.findall(r'\b\w+\b', str(text).lower())
            unique_words_in_review = {word for word in words if word not in custom_stopwords and len(word) > 2}
            mention_counter.update(unique_words_in_review)
        return mention_counter.most_common(n)
    top_keywords = get_top_keywords_by_mention(chart_data["text"])
    formatted_options = [f"{word} ({count} mentions)" for word, count in top_keywords]
    selected_option = st.selectbox("Select a keyword to analyze:", options=["--- Select a Keyword ---"] + formatted_options, on_change=reset_keyword_page)
    if selected_option != "--- Select a Keyword ---":
        selected_keyword = selected_option.split(' ')[0]
        keyword_df = chart_data[chart_data['text'].str.contains(r'\b' + selected_keyword + r'\b', case=False, na=False)]
        st.markdown(f"---")
        st.markdown(f"#### Analysis for keyword: `{selected_keyword}` ({len(keyword_df)} mentions)")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Rating Distribution**")
            rating_dist = keyword_df['rating'].value_counts().reindex(range(1, 6), fill_value=0).sort_index()
            st.bar_chart(rating_dist)
        with col2:
            st.markdown("**Sentiment Distribution**")
            sentiment_dist = keyword_df['sentiment'].value_counts().reindex(['Positive', 'Neutral', 'Negative'], fill_value=0)
            st.bar_chart(sentiment_dist)
        st.markdown("**Example Reviews**")
        sort_reviews_by = st.selectbox("Sort examples by:",("Most Helpful", "Newest", "Oldest", "Highest Rating", "Lowest Rating"),key="keyword_review_sort",on_change=reset_keyword_page)
        if sort_reviews_by == "Most Helpful":
            sorted_keyword_df = keyword_df.sort_values(by="helpful_vote", ascending=False)
        elif sort_reviews_by == "Highest Rating":
            sorted_keyword_df = keyword_df.sort_values(by="rating", ascending=False)
        elif sort_reviews_by == "Lowest Rating":
            sorted_keyword_df = keyword_df.sort_values(by="rating", ascending=True)
        elif sort_reviews_by == "Oldest":
            sorted_keyword_df = keyword_df.sort_values(by="date", ascending=True)
        else:
            sorted_keyword_df = keyword_df.sort_values(by="date", ascending=False)
        if 'keyword_review_page' not in st.session_state:
            st.session_state.keyword_review_page = 0
        start_idx = st.session_state.keyword_review_page * REVIEWS_PER_PAGE
        end_idx = start_idx + REVIEWS_PER_PAGE
        reviews_to_display = sorted_keyword_df.iloc[start_idx:end_idx]
        for _, review in reviews_to_display.iterrows():
            with st.container(border=True):
                st.subheader(review['review_title'])
                caption_parts = []
                if review['verified_purchase']:
                    caption_parts.append("✅ Verified")
                caption_parts.append(f"Reviewed on: {review['date']}")
                caption_parts.append(f"Rating: {review['rating']} ⭐")
                caption_parts.append(f"Helpful Votes: {review['helpful_vote']} 👍")
                st.caption(" | ".join(caption_parts))
                st.markdown(f"> {review['text']}")
        total_reviews = len(sorted_keyword_df)
        total_pages = (total_reviews + REVIEWS_PER_PAGE - 1) // REVIEWS_PER_PAGE
        if total_pages > 1:
            st.caption(f"Page {st.session_state.keyword_review_page + 1} of {total_pages}")
            p_col1, p_col2 = st.columns(2)
            with p_col1:
                if st.session_state.keyword_review_page > 0:
                    if st.button("⬅️ Previous 5 Reviews"):
                        st.session_state.keyword_review_page -= 1
                        st.rerun()
            with p_col2:
                if end_idx < total_reviews:
                    if st.button("Next 5 Reviews ➡️"):
                        st.session_state.keyword_review_page += 1
                        st.rerun()

if __name__ == "__main__":
    main()
