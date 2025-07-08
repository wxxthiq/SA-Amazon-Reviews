# pages/4_Aspect_Analysis.py
import streamlit as st
import pandas as pd
import spacy
from collections import Counter
import re
from textblob import TextBlob
from utils.database_utils import (
    connect_to_db,
    get_product_details,
    get_reviews_for_product,
    get_product_date_range
)

# --- Page Configuration and NLP Model Loading ---
st.set_page_config(layout="wide", page_title="Aspect Analysis")

@st.cache_resource
def load_spacy_model():
    # Load the small English model
    return spacy.load("en_core_web_sm")

nlp = load_spacy_model()
DB_PATH = "amazon_reviews_top100.duckdb"
conn = connect_to_db(DB_PATH)
REVIEWS_PER_PAGE = 5

# --- Main App Logic ---
def main():
    st.title("🔎 Automated Aspect Analysis")

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
    st.caption("This page automatically identifies the most talked-about features (aspects) in the reviews.")

    # --- Sidebar Filters ---
    st.sidebar.header("🔬 Aspect Analysis Filters")
    min_date_db, max_date_db = get_product_date_range(conn, selected_asin)
    
    default_date_range = (min_date_db, max_date_db)
    default_ratings = [1, 2, 3, 4, 5]
    default_verified = "All"
    
    selected_date_range = st.sidebar.date_input("Filter by Date Range", value=default_date_range, key='aspect_date_filter')
    selected_ratings = st.sidebar.multiselect("Filter by Star Rating", options=default_ratings, default=default_ratings, key='aspect_rating_filter')
    selected_verified = st.sidebar.radio("Filter by Purchase Status", ["All", "Verified Only", "Not Verified"], index=0, key='aspect_verified_filter')
    
    # Load data based on the filters
    chart_data = get_reviews_for_product(conn, selected_asin, selected_date_range, tuple(selected_ratings), ['Positive', 'Negative', 'Neutral'], selected_verified)

    st.markdown("---")
    if chart_data.empty:
        st.warning("No review data available for the selected filters.")
        st.stop()
        
    st.info(f"Analyzing aspects from **{len(chart_data)}** reviews matching your criteria.")

    # --- AUTOMATED ASPECT EXTRACTION ---
    @st.cache_data
    def extract_top_aspects(texts, top_n=15):
        # We will identify aspects as the most common nouns in the reviews.
        all_nouns = []
        # Process documents in batches for efficiency
        for doc in nlp.pipe(texts, disable=["parser", "ner"]):
            # Use NOUN and PROPN (proper nouns)
            all_nouns.extend([token.lemma_.lower() for token in doc if token.pos_ in ('NOUN', 'PROPN') and len(token.lemma_) > 2])
        
        # Count the frequency of each noun
        noun_freq = Counter(all_nouns)
        return [noun for noun, freq in noun_freq.most_common(top_n)]

    with st.spinner("Automatically identifying key aspects from reviews..."):
        top_aspects = extract_top_aspects(chart_data['text'].astype(str))

    st.markdown("### 🔬 Interactive Aspect Explorer")
    selected_aspect = st.selectbox(
        "Select an auto-detected aspect to analyze:",
        options=["--- Select an Aspect ---"] + top_aspects
    )

    if selected_aspect != "--- Select an Aspect ---":
        # --- On-the-Fly Aspect Sentiment Calculation ---
        @st.cache_data
        def calculate_aspect_sentiments(data, aspect, window=10):
            sentiments = []
            snippets = []
            
            for index, row in data.iterrows():
                text = str(row['text']).lower()
                if re.search(r'\b' + aspect + r'\b', text):
                    for match in re.finditer(r'\b' + aspect + r'\b', text):
                        start, end = match.start(), match.end()
                        words_before = text[:start].split()[-window:]
                        words_after = text[end:].split()[:window]
                        context_text = " ".join(words_before + [aspect] + words_after)
                        
                        sentiment = TextBlob(context_text).sentiment.polarity
                        sentiments.append(sentiment)
                        
                        highlighted_snippet = " ".join(words_before) + f" **:orange[{aspect}]** " + " ".join(words_after)
                        snippets.append((row['review_id'], highlighted_snippet, row['rating'], row['helpful_vote']))
            
            return sentiments, snippets

        aspect_sentiments, aspect_snippets = calculate_aspect_sentiments(chart_data, selected_aspect)

        st.markdown(f"---")
        st.markdown(f"#### Analysis for aspect: `{selected_aspect}` ({len(aspect_sentiments)} mentions)")
        
        if not aspect_sentiments:
            st.warning(f"No mentions of the aspect '{selected_aspect}' found in the filtered reviews.")
        else:
            sentiment_df = pd.DataFrame(aspect_sentiments, columns=['polarity'])
            sentiment_df['sentiment_category'] = sentiment_df['polarity'].apply(lambda p: 'Positive' if p > 0.1 else ('Negative' if p < -0.1 else 'Neutral'))
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Aspect Sentiment Distribution**")
                dist = sentiment_df['sentiment_category'].value_counts().reindex(['Positive', 'Neutral', 'Negative'], fill_value=0)
                st.bar_chart(dist)
            with col2:
                st.markdown("**Sentiment Polarity Trend**")
                st.line_chart(sentiment_df['polarity'])

            st.markdown(f"**Example mentions of `{selected_aspect}`**")
            # Sort snippets by helpfulness to show the most impactful examples first
            sorted_snippets = sorted(aspect_snippets, key=lambda x: x[3], reverse=True)

            for review_id, snippet, rating, helpful_votes in sorted_snippets[:10]:
                with st.container(border=True):
                    st.caption(f"From a {rating} ⭐ review ({helpful_votes} helpful votes)")
                    st.markdown(f"...{snippet}...")

if __name__ == "__main__":
    main()
