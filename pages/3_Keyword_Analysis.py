# pages/3_Keyword_Analysis.py
import streamlit as st
import pandas as pd
import numpy as np
import re
from collections import Counter
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
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
    st.title("🔑 Detailed Keyword & Phrase Analysis")

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
    st.caption("Use the sidebar to filter reviews, then explore the most common terms and phrases.")

    # --- DEDICATED SIDEBAR FILTERS FOR THIS PAGE ---
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

    # --- N-GRAM WORD CLOUD SUMMARY ---
    st.markdown("### ☁️ Keyword & Phrase Summary")
    
    col1, col2 = st.columns([1,1])
    with col1:
        max_words = st.slider("Max Terms in Cloud:", min_value=5, max_value=50, value=15, key="keyword_cloud_slider")
    with col2:
        ngram_level = st.radio("Term Type:", ("Single Words", "Bigrams", "Trigrams"), index=0, horizontal=True, key="keyword_ngram_radio")
    
    def get_top_ngrams(corpus, n=None, ngram_range=(1,1)):
        vec = CountVectorizer(ngram_range=ngram_range, stop_words='english').fit(corpus)
        bag_of_words = vec.transform(corpus)
        sum_words = bag_of_words.sum(axis=0) 
        words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
        words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
        return words_freq[:n]

    ngram_range = {
        "Single Words": (1,1),
        "Bigrams": (2,2),
        "Trigrams": (3,3)
    }.get(ngram_level, (1,1))

    wc_col1, wc_col2 = st.columns(2)
    with wc_col1:
        st.markdown("#### Positive Terms")
        pos_text = chart_data[chart_data["sentiment"]=="Positive"]["text"].dropna()
        if not pos_text.empty:
            top_pos_grams = get_top_ngrams(pos_text, n=max_words, ngram_range=ngram_range)
            pos_freq_dict = dict(top_pos_grams)
            if pos_freq_dict:
                wordcloud_pos = WordCloud(stopwords=STOPWORDS, background_color="white", width=800, height=400, colormap='Greens').generate_from_frequencies(pos_freq_dict)
                fig, ax = plt.subplots()
                ax.imshow(wordcloud_pos, interpolation='bilinear')
                ax.axis("off")
                st.pyplot(fig)
    with wc_col2:
        st.markdown("#### Negative Terms")
        neg_text = chart_data[chart_data["sentiment"]=="Negative"]["text"].dropna()
        if not neg_text.empty:
            top_neg_grams = get_top_ngrams(neg_text, n=max_words, ngram_range=ngram_range)
            neg_freq_dict = dict(top_neg_grams)
            if neg_freq_dict:
                wordcloud_neg = WordCloud(stopwords=STOPWORDS, background_color="white", width=800, height=400, colormap='Reds').generate_from_frequencies(neg_freq_dict)
                fig, ax = plt.subplots()
                ax.imshow(wordcloud_neg, interpolation='bilinear')
                ax.axis("off")
                st.pyplot(fig)

    # --- INTERACTIVE KEYWORD EXPLORER ---
    st.markdown("---")
    st.markdown("### 🔬 Interactive Term Explorer")

    @st.cache_data
    def get_top_ngrams_by_mention(corpus, n=25, ngram_range=(1,1)):
        vec = CountVectorizer(ngram_range=ngram_range, stop_words='english').fit(corpus)
        bag_of_words = vec.transform(corpus)
        terms = vec.get_feature_names_out()
        term_mentions = np.asarray(bag_of_words.sum(axis=0)).flatten()
        term_freq = [(term, term_mentions[i]) for i, term in enumerate(terms)]
        term_freq = sorted(term_freq, key=lambda x: x[1], reverse=True)
        return term_freq[:n]

    top_terms = get_top_ngrams_by_mention(chart_data["text"].dropna(), n=25, ngram_range=ngram_range)
    formatted_options = [f"{term} ({count} mentions)" for term, count in top_terms]
    
    selected_option = st.selectbox(
        "Select a term to analyze:",
        options=["--- Select a Term ---"] + formatted_options,
        on_change=reset_keyword_page
    )

    if selected_option != "--- Select a Term ---":
        match = re.match(r"(.+) \((\d+) mentions\)", selected_option)
        if match:
            selected_term = match.group(1)
            
            keyword_df = chart_data[chart_data['text'].str.contains(re.escape(selected_term), case=False, na=False)]
            
            st.markdown(f"---")
            st.markdown(f"#### Analysis for term: `{selected_term}` ({len(keyword_df)} mentions)")

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
            sort_reviews_by = st.selectbox(
                "Sort examples by:",
                ("Most Helpful", "Newest", "Oldest", "Highest Rating", "Lowest Rating"),
                key="keyword_review_sort",
                on_change=reset_keyword_page
            )
            
            # ** KEY CHANGE: Added secondary sort key for rating options **
            if sort_reviews_by == "Most Helpful":
                sorted_keyword_df = keyword_df.sort_values(by="helpful_vote", ascending=False)
            elif sort_reviews_by == "Highest Rating":
                sorted_keyword_df = keyword_df.sort_values(by=["rating", "helpful_vote"], ascending=[False, False])
            elif sort_reviews_by == "Lowest Rating":
                sorted_keyword_df = keyword_df.sort_values(by=["rating", "helpful_vote"], ascending=[True, False])
            elif sort_reviews_by == "Oldest":
                sorted_keyword_df = keyword_df.sort_values(by="date", ascending=True)
            else: # Newest
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
