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
    st.caption("This page automatically identifies key product features and analyzes the specific sentiment towards them.")

    # --- Sidebar Filters (COMPLETE SET) ---
    st.sidebar.header("🔬 Aspect Analysis Filters")
    min_date_db, max_date_db = get_product_date_range(conn, selected_asin)
    
    default_date_range = (min_date_db, max_date_db)
    default_ratings = [1, 2, 3, 4, 5]
    default_sentiments = ['Positive', 'Negative', 'Neutral'] # Added sentiment filter
    default_verified = "All"

    # Callback to reset all filters on this page
    def reset_all_aspect_filters():
        st.session_state.aspect_date_filter = default_date_range
        st.session_state.aspect_rating_filter = default_ratings
        st.session_state.aspect_sentiment_filter = default_sentiments
        st.session_state.aspect_verified_filter = default_verified
    
    selected_date_range = st.sidebar.date_input("Filter by Date Range", value=default_date_range, key='aspect_date_filter')
    selected_ratings = st.sidebar.multiselect("Filter by Star Rating", options=default_ratings, default=default_ratings, key='aspect_rating_filter')
    selected_sentiments = st.sidebar.multiselect("Filter by Sentiment", options=default_sentiments, default=default_sentiments, key='aspect_sentiment_filter') # Added widget
    selected_verified = st.sidebar.radio("Filter by Purchase Status", ["All", "Verified Only", "Not Verified"], index=0, key='aspect_verified_filter')
    
    # Add the reset button
    st.sidebar.button("Reset All Filters", on_click=reset_all_aspect_filters, use_container_width=True, key='reset_aspect_filters')
    # Load data based on all filters
    chart_data = get_reviews_for_product(conn, selected_asin, selected_date_range, tuple(selected_ratings), tuple(selected_sentiments), selected_verified)

    st.markdown("---")
    if chart_data.empty:
        st.warning("No review data available for the selected filters.")
        st.stop()
        
    st.info(f"Analyzing aspects from **{len(chart_data)}** reviews matching your criteria.")

    # --- Aspect Summary Chart (Unchanged) ---
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

    # --- Interactive Aspect Explorer (ENHANCED) ---
    st.markdown("---")
    st.markdown("### 🔬 Interactive Aspect Explorer")
    selected_aspect = st.selectbox(
        "Select an auto-detected aspect to analyze in detail:",
        options=["--- Select an Aspect ---"] + top_aspects_list
    )

    if selected_aspect != "--- Select an Aspect ---":
        # Find all reviews that mention the selected aspect
        aspect_df = chart_data[chart_data['text'].str.contains(r'\b' + re.escape(selected_aspect) + r'\b', case=False, na=False)].copy()
        
        st.markdown(f"---")
        st.markdown(f"#### Analysis for aspect: `{selected_aspect}` ({len(aspect_df)} mentions)")
        
        if aspect_df.empty:
            st.warning(f"No mentions of '{selected_aspect}' found with the current filters.")
        else:
            # --- Distribution Charts for the Aspect ---
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Rating Distribution for this Aspect**")
                rating_dist = aspect_df['rating'].value_counts().reindex(range(1, 6), fill_value=0).sort_index()
                st.bar_chart(rating_dist)
            with col2:
                st.markdown("**Sentiment Distribution for this Aspect**")
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
                    rating_stream_chart = px.area(rating_counts_over_time, x='period', y='count', color='rating', color_discrete_map={5: '#1a9850', 4: '#91cf60', 3: '#d9ef8b', 2: '#fee08b', 1: '#d73027'}, category_orders={"rating": [5, 4, 3, 2, 1])
                    st.plotly_chart(rating_stream_chart, use_container_width=True)
            with t_col2:
                st.markdown("###### Sentiment Volume")
                sentiment_counts_over_time = time_df.groupby(['period', 'sentiment']).size().reset_index(name='count')
                if not sentiment_counts_over_time.empty:
                    sentiment_stream_chart = px.area(sentiment_counts_over_time, x='period', y='count', color='sentiment', color_discrete_map={'Positive': '#1a9850', 'Neutral': '#cccccc', 'Negative': '#d73027'}, category_orders={"sentiment": ["Positive", "Neutral", "Negative"])
                    st.plotly_chart(sentiment_stream_chart, use_container_width=True)

            # --- Example Reviews Display ---
            st.markdown("---")
            st.markdown("**Example Reviews Mentioning this Aspect**")

            # ** KEY CHANGE: Using a lambda function for robust replacement **
            def highlight_text(text, aspect):
                # This lambda function is a safer way to perform the replacement
                return re.sub(
                    r'(\b' + re.escape(aspect) + r'\b)',
                    lambda m: f'**<span style="color:orange">{m.group(1)}</span>**',
                    text,
                    flags=re.IGNORECASE
                )

            # Sort and display reviews
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
                    
                    # Highlight the aspect in the full review text
                    highlighted_review = highlight_text(review['text'], selected_aspect)
                    st.markdown(f"> {highlighted_review}", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
