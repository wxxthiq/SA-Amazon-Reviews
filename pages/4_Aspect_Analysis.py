# pages/4_Aspect_Analysis.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import altair as alt
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
    return spacy.load("en_core_web_sm")

nlp = load_spacy_model()
DB_PATH = "amazon_reviews_top100.duckdb"
conn = connect_to_db(DB_PATH)

# --- Main App Logic ---
def main():
    st.title("🔎 Aspect-Based Sentiment Analysis")

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
    st.caption("This page automatically identifies key product features (aspects) and analyzes the specific sentiment towards them.")

    # --- Sidebar Filters (COMPLETE SET) ---
    st.sidebar.header("🔬 Aspect Analysis Filters")
    min_date_db, max_date_db = get_product_date_range(conn, selected_asin)
    
    default_date_range = (min_date_db, max_date_db)
    default_ratings = [1, 2, 3, 4, 5]
    default_sentiments = ['Positive', 'Negative', 'Neutral']
    default_verified = "All"
    
    selected_date_range = st.sidebar.date_input("Filter by Date Range", value=default_date_range, key='aspect_date_filter')
    selected_ratings = st.sidebar.multiselect("Filter by Star Rating", options=default_ratings, default=default_ratings, key='aspect_rating_filter')
    selected_sentiments = st.sidebar.multiselect("Filter by Sentiment", options=default_sentiments, default=default_sentiments, key='aspect_sentiment_filter')
    selected_verified = st.sidebar.radio("Filter by Purchase Status", ["All", "Verified Only", "Not Verified"], index=0, key='aspect_verified_filter')
    
    # Load data based on all filters
    chart_data = get_reviews_for_product(conn, selected_asin, selected_date_range, tuple(selected_ratings), tuple(selected_sentiments), selected_verified)

    st.markdown("---")
    if chart_data.empty:
        st.warning("No review data available for the selected filters.")
        st.stop()
        
    st.info(f"Analyzing aspects from **{len(chart_data)}** reviews matching your criteria.")

    # --- Automated Aspect Extraction ---
    @st.cache_data
    def extract_top_aspects(texts, top_n=15):
        all_aspects = []
        for doc in nlp.pipe(texts, disable=["ner"]): # Keep the parser enabled
            for chunk in doc.noun_chunks:
                # A simple filter to avoid very short or irrelevant chunks
                if len(chunk.text.split()) > 1 and chunk.root.pos_ != 'PRON':
                    all_aspects.append(chunk.lemma_.lower())
        return [aspect for aspect, freq in Counter(all_aspects).most_common(top_n)]

    with st.spinner("Automatically identifying key aspects from reviews..."):
        top_aspects = extract_top_aspects(chart_data['text'].astype(str))

    # --- Interactive Aspect Explorer ---
    st.markdown("### 🔬 Interactive Aspect Explorer")
    selected_aspect = st.selectbox(
        "Select an auto-detected aspect to analyze in detail:",
        options=["--- Select an Aspect ---"] + top_aspects
    )

    if selected_aspect != "--- Select an Aspect ---":
        # Filter the DataFrame for reviews mentioning the aspect
        aspect_df = chart_data[chart_data['text'].str.contains(r'\b' + re.escape(selected_aspect) + r'\b', case=False, na=False)].copy()

        st.markdown(f"---")
        st.markdown(f"#### Analysis for aspect: `{selected_aspect}` ({len(aspect_df)} mentions)")
        
        if aspect_df.empty:
            st.warning(f"No mentions of '{selected_aspect}' found with the current filters.")
        else:
            # --- Distribution Charts for the Aspect ---
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Rating Distribution**")
                rating_dist = aspect_df['rating'].value_counts().reindex(range(1, 6), fill_value=0).sort_index()
                st.bar_chart(rating_dist)
            with col2:
                st.markdown("**Sentiment Distribution**")
                sentiment_dist = aspect_df['sentiment'].value_counts().reindex(['Positive', 'Neutral', 'Negative'], fill_value=0)
                st.bar_chart(sentiment_dist)
            
            # --- Trend Charts for the Aspect ---
            st.markdown("---")
            st.markdown("**Trends for this Aspect Over Time**")
            
            time_df = aspect_df.copy()
            time_df['date'] = pd.to_datetime(time_df['date'])
            time_df['period'] = time_df['date'].dt.to_period('M').dt.start_time
            
            t_col1, t_col2 = st.columns(2)
            with t_col1:
                st.markdown("###### Rating Volume")
                rating_counts_over_time = time_df.groupby(['period', 'rating']).size().reset_index(name='count')
                if not rating_counts_over_time.empty:
                    rating_stream_chart = px.area(rating_counts_over_time, x='period', y='count', color='rating', color_discrete_map={5: '#1a9850', 4: '#91cf60', 3: '#d9ef8b', 2: '#fee08b', 1: '#d73027'})
                    st.plotly_chart(rating_stream_chart, use_container_width=True)
            with t_col2:
                st.markdown("###### Sentiment Volume")
                sentiment_counts_over_time = time_df.groupby(['period', 'sentiment']).size().reset_index(name='count')
                if not sentiment_counts_over_time.empty:
                    sentiment_stream_chart = px.area(sentiment_counts_over_time, x='period', y='count', color='sentiment', color_discrete_map={'Positive': '#1a9850', 'Neutral': '#cccccc', 'Negative': '#d73027'})
                    st.plotly_chart(sentiment_stream_chart, use_container_width=True)

            # --- Example Reviews Display ---
            st.markdown("---")
            st.markdown("**Example Reviews Mentioning this Aspect**")

            def highlight_text(text, aspect):
                return re.sub(r'(\b' + re.escape(aspect) + r'\b)', r'**:\orange[\1]**', text, flags=re.IGNORECASE)

            sorted_aspect_df = aspect_df.sort_values(by="helpful_vote", ascending=False)
            
            for _, review in sorted_aspect_df.head(10).iterrows():
                with st.container(border=True):
                    st.subheader(review['review_title'])
                    
                    caption_parts = []
                    if review['verified_purchase']:
                        caption_parts.append("✅ Verified")
                    caption_parts.append(f"Reviewed on: {review['date']}")
                    caption_parts.append(f"Rating: {review['rating']} ⭐")
                    caption_parts.append(f"Helpful Votes: {review['helpful_vote']} 👍")
                    st.caption(" | ".join(caption_parts))
                    
                    highlighted_review = highlight_text(review['text'], selected_aspect)
                    st.markdown(f"> {highlighted_review}", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
