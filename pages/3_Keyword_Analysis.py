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
    st.caption("Use the sidebar to filter the reviews, then select a keyword to analyze.")

   # --- DEDICATED SIDEBAR FILTERS FOR THIS PAGE ---
    st.sidebar.header("🔬 Keyword Analysis Filters")
    min_date_db, max_date_db = get_product_date_range(conn, selected_asin)
    
    default_date_range = (min_date_db, max_date_db)
    default_ratings = [1, 2, 3, 4, 5]
    default_sentiments = ['Positive', 'Negative', 'Neutral']
    default_verified = "All"
    
    selected_date_range = st.sidebar.date_input("Filter by Date Range", value=default_date_range, key='keyword_date_filter')
    selected_ratings = st.sidebar.multiselect("Filter by Star Rating", options=default_ratings, default=default_ratings, key='keyword_rating_filter')
    selected_sentiments = st.sidebar.multiselect("Filter by Sentiment", options=default_sentiments, default=default_sentiments, key='keyword_sentiment_filter')
    # ** NEW: Verified Purchase Filter **
    selected_verified = st.sidebar.radio("Filter by Purchase Status", ["All", "Verified Only", "Not Verified"], index=0, key='keyword_verified_filter')
    
    # Load data based on the local filters
    chart_data = get_reviews_for_product(conn, selected_asin, selected_date_range, tuple(selected_ratings), tuple(selected_sentiments), selected_verified)

    st.markdown("---")
    if chart_data.empty:
        st.warning("No review data available for the selected filters.")
        st.stop()
        
    st.info(f"Analyzing keywords from **{len(chart_data)}** reviews matching your criteria.")

    # --- WORD CLOUD SUMMARY ---
    st.markdown("### ☁️ Keyword Summary")
    
    wc_col1, wc_col2 = st.columns(2)
    with wc_col1:
        st.markdown("#### Positive Keywords")
        pos_text = " ".join(review for review in chart_data[chart_data["sentiment"]=="Positive"]["text"])
        if pos_text:
            wordcloud_pos = WordCloud(stopwords=STOPWORDS, background_color="white", width=800, height=400, colormap='Greens').generate(pos_text)
            fig, ax = plt.subplots()
            ax.imshow(wordcloud_pos, interpolation='bilinear')
            ax.axis("off")
            st.pyplot(fig)
        else:
            st.info("No positive reviews to generate a word cloud.")

    with wc_col2:
        st.markdown("#### Negative Keywords")
        neg_text = " ".join(review for review in chart_data[chart_data["sentiment"]=="Negative"]["text"])
        if neg_text:
            wordcloud_neg = WordCloud(stopwords=STOPWORDS, background_color="white", width=800, height=400, colormap='Reds').generate(neg_text)
            fig, ax = plt.subplots()
            ax.imshow(wordcloud_neg, interpolation='bilinear')
            ax.axis("off")
            st.pyplot(fig)
        else:
            st.info("No negative reviews to generate a word cloud.")

    # --- INTERACTIVE KEYWORD EXPLORER ---
    st.markdown("---")
    st.markdown("### 🔬 Interactive Keyword Explorer")

    # Helper function to get top keywords
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
