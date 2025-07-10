# pages/5_Product_Comparison.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from utils.database_utils import (
    connect_to_db,
    get_product_details,
    get_reviews_for_product
)
import altair as alt
from textblob import TextBlob
from collections import Counter
import re
import spacy

# --- Page Configuration and Model Loading ---
st.set_page_config(layout="wide", page_title="Product Comparison")
DB_PATH = "amazon_reviews_top100.duckdb"
conn = connect_to_db(DB_PATH)

@st.cache_resource
def load_spacy_model():
    return spacy.load("en_core_web_sm")

nlp = load_spacy_model()

# --- Helper function for Aspect Analysis ---
@st.cache_data
def get_aspect_summary_for_comparison(_data, num_aspects=5):
    all_aspects = []
    def clean_chunk(chunk):
        return " ".join(token.lemma_.lower() for token in chunk if token.pos_ in ['NOUN', 'PROPN', 'ADJ'])

    for doc in nlp.pipe(_data['text'].astype(str)):
        for chunk in doc.noun_chunks:
            cleaned = clean_chunk(chunk)
            if cleaned and len(cleaned) > 2:
                all_aspects.append(cleaned)
    
    if not all_aspects:
        return pd.DataFrame()
        
    top_aspects = [aspect for aspect, freq in Counter(all_aspects).most_common(num_aspects)]
    
    aspect_sentiments = []
    for aspect in top_aspects:
        for text in _data['text']:
            if re.search(r'\b' + re.escape(aspect) + r'\b', str(text).lower()):
                window = str(text).lower()[max(0, str(text).lower().find(aspect)-50):min(len(text), str(text).lower().find(aspect)+len(aspect)+50)]
                polarity = TextBlob(window).sentiment.polarity
                sentiment_cat = 'Positive' if polarity > 0.1 else 'Negative' if polarity < -0.1 else 'Neutral'
                aspect_sentiments.append({'aspect': aspect, 'sentiment': sentiment_cat})
    
    if not aspect_sentiments:
        return pd.DataFrame()
        
    return pd.DataFrame(aspect_sentiments)


# --- Main App Logic ---
def main():
    st.title("⚖️ Product Comparison")

    if st.button("⬅️ Back to Search"):
        st.switch_page("app.py")

    # Check if there are products selected for comparison
    if 'products_to_compare' not in st.session_state or not st.session_state.products_to_compare:
        st.warning("Please select 2 to 4 products from the main search page to compare.")
        st.stop()

    selected_asins = st.session_state.products_to_compare
    
    if len(selected_asins) < 2:
        st.warning("Please select at least two products to compare.")
        st.stop()
        
    st.info(f"Comparing **{len(selected_asins)}** products. Use the sidebar to apply universal filters to all products.")

    # --- Sidebar for Universal Filters ---
    st.sidebar.header("📊 Universal Comparison Filters")
    # For simplicity, we'll start with just a verified purchase filter
    # You can add date, rating, etc. here later if needed
    verified_filter = st.sidebar.radio(
        "Filter by Purchase Status", 
        ["All", "Verified Only", "Not Verified"], 
        index=0, 
        key='comparison_verified_filter'
    )

    # --- Fetch and Display Products ---
    product_data_cache = {}
    review_data_cache = {}

    with st.spinner("Loading product data..."):
        for asin in selected_asins:
            product_details = get_product_details(conn, asin)
            if not product_details.empty:
                product_data_cache[asin] = product_details.iloc[0]
                # Fetch all reviews initially based on the filter
                review_data_cache[asin] = get_reviews_for_product(
                    conn, asin, date_range=(), rating_filter=(), sentiment_filter=(), verified_filter=verified_filter
                )

    cols = st.columns(len(selected_asins))

    # --- Section 1: Basic Info ---
    st.markdown("---")
    st.header("General Information")
    for i, asin in enumerate(selected_asins):
        with cols[i]:
            if asin in product_data_cache:
                product = product_data_cache[asin]
                st.subheader(product['product_title'])
                
                image_urls_str = product.get('image_urls')
                image_url = image_urls_str.split(',')[0] if pd.notna(image_urls_str) else "https://via.placeholder.com/200"
                st.image(image_url, use_container_width=True)
                
                st.metric("Average Rating", f"{product.get('average_rating', 0):.2f} ⭐")
                st.metric("Total Reviews", f"{int(product.get('review_count', 0)):,}")
                st.caption(f"Category: {product['category']}")

    # --- Section 2: Rating & Sentiment Distribution ---
    st.markdown("---")
    st.header("Rating & Sentiment Distribution")
    rating_cols = st.columns(len(selected_asins))
    
    for i, asin in enumerate(selected_asins):
        with rating_cols[i]:
            if asin in review_data_cache and not review_data_cache[asin].empty:
                reviews_df = review_data_cache[asin]
                st.markdown("**Rating Distribution**")
                rating_counts = reviews_df['rating'].value_counts().reindex(range(1, 6), fill_value=0)
                st.bar_chart(rating_counts)

                st.markdown("**Sentiment Distribution**")
                sentiment_counts = reviews_df['sentiment'].value_counts().reindex(['Positive', 'Neutral', 'Negative'], fill_value=0)
                st.bar_chart(sentiment_counts)
            else:
                st.info("No review data for this product.")

    # --- Section 3: Aspect Sentiment Comparison ---
    st.markdown("---")
    st.header("Aspect Sentiment Radar")
    st.caption("Compares sentiment towards the top 5 most common aspects for each product.")
    
    radar_cols = st.columns(len(selected_asins))
    for i, asin in enumerate(selected_asins):
        with radar_cols[i]:
             if asin in review_data_cache and not review_data_cache[asin].empty:
                reviews_df = review_data_cache[asin]
                aspect_summary_df = get_aspect_summary_for_comparison(reviews_df)

                if not aspect_summary_df.empty:
                    radar_df = aspect_summary_df.groupby(['aspect', 'sentiment']).size().unstack(fill_value=0)
                    categories = ['Positive', 'Negative', 'Neutral']
                    for sent in categories:
                        if sent not in radar_df.columns:
                            radar_df[sent] = 0
                    radar_df = radar_df[categories]

                    fig = go.Figure()
                    for aspect in radar_df.index:
                        fig.add_trace(go.Scatterpolar(
                            r=radar_df.loc[aspect].values,
                            theta=categories,
                            fill='toself',
                            name=aspect
                        ))
                    
                    fig.update_layout(
                        polar=dict(radialaxis=dict(visible=True, range=[0, radar_df.max().max()])),
                        showlegend=True,
                        title=f"Aspects for {product_data_cache[asin]['product_title'][:30]}..."
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Not enough data for aspect analysis.")

if __name__ == "__main__":
    main()
