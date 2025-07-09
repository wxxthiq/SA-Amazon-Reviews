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

    # --- Aspect Summary Chart ---
    @st.cache_data
    def get_aspect_summary(data):
        all_aspects = []
        def clean_chunk(chunk):
            return " ".join(token.lemma_.lower() for token in chunk if token.pos_ in ['NOUN', 'PROPN', 'ADJ'])
        
        for doc in nlp.pipe(data['text'].astype(str)):
            for chunk in doc.noun_chunks:
                cleaned = clean_chunk(chunk)
                if cleaned and len(cleaned) > 2:
                    all_aspects.append(cleaned)
        
        if not all_aspects:
            return pd.DataFrame(), []
            
        top_aspects = [aspect for aspect, freq in Counter(all_aspects).most_common(5)]
        
        aspect_sentiments = []
        for aspect in top_aspects:
            for text in data['text']:
                if re.search(r'\b' + re.escape(aspect) + r'\b', str(text).lower()):
                    window = str(text).lower()[max(0, str(text).lower().find(aspect)-50):min(len(text), str(text).lower().find(aspect)+len(aspect)+50)]
                    polarity = TextBlob(window).sentiment.polarity
                    sentiment_cat = 'Positive' if polarity > 0.1 else 'Negative' if polarity < -0.1 else 'Neutral'
                    aspect_sentiments.append({'aspect': aspect, 'sentiment': sentiment_cat})
        
        return pd.DataFrame(aspect_sentiments), top_aspects

    st.markdown("### Aspect Sentiment Summary")
    aspect_summary_df, top_aspects_list = get_aspect_summary(chart_data)

    if not aspect_summary_df.empty:
        summary_chart_data = aspect_summary_df.groupby(['aspect', 'sentiment']).size().reset_index(name='count')
        chart = alt.Chart(summary_chart_data).mark_bar().encode(
            x=alt.X('count:Q', title='Number of Mentions'),
            y=alt.Y('aspect:N', sort='-x', title='Aspect'),
            color=alt.Color('sentiment:N', scale=alt.Scale(domain=['Positive', 'Neutral', 'Negative'], range=['#1a9850', '#cccccc', '#d73027']), legend=alt.Legend(title="Sentiment")),
            yOffset='sentiment:N'
        ).configure_axis(grid=False).configure_view(strokeWidth=0)
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("Not enough data to generate an aspect summary.")

    # --- Interactive Aspect Explorer ---
    st.markdown("---")
    st.markdown("### 🔬 Interactive Aspect Explorer")
    selected_aspect = st.selectbox(
        "Select an auto-detected aspect to analyze in detail:",
        options=["--- Select an Aspect ---"] + top_aspects_list
    )

    if selected_aspect != "--- Select an Aspect ---":
        @st.cache_data
        def calculate_aspect_sentiments(data, aspect, window=10):
            sentiments = []
            snippets = []
            
            for index, row in data.iterrows():
                text = str(row['text']).lower()
                # Use regex to find the aspect as a whole word
                if re.search(r'\b' + re.escape(aspect) + r'\b', text):
                    for match in re.finditer(r'\b' + re.escape(aspect) + r'\b', text):
                        start, end = match.start(), match.end()
                        words_before = text[:start].split()[-window:]
                        words_after = text[end:].split()[:window]
                        context_text = " ".join(words_before + [aspect] + words_after)
                        
                        sentiment = TextBlob(context_text).sentiment.polarity
                        sentiments.append(sentiment)
                        
                        # Create a highlighted snippet for display
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
            sorted_snippets = sorted(aspect_snippets, key=lambda x: x[3], reverse=True)

            for review_id, snippet, rating, helpful_votes in sorted_snippets[:10]:
                with st.container(border=True):
                    st.caption(f"From a {rating} ⭐ review ({helpful_votes} helpful votes)")
                    st.markdown(f"...{snippet}...")

if __name__ == "__main__":
    main()
