# pages/5_Product_Comparison.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import spacy
import re
import json
from utils.database_utils import (
    connect_to_db,
    get_product_details,
    get_reviews_for_product,
    get_all_categories,
    get_filtered_products 
)

# --- Page Config & Model Loading ---
st.set_page_config(layout="wide", page_title="Product Comparison")

@st.cache_resource
def load_spacy_model():
    return spacy.load("en_core_web_sm")

nlp = load_spacy_model()
DB_PATH = "amazon_reviews_final.duckdb"
conn = connect_to_db(DB_PATH)

# --- Aspect Extraction Function (consistent with overview page) ---
@st.cache_data
def extract_aspects_with_sentiment(dataf):
    aspect_sentiments = []
    stop_words = nlp.Defaults.stop_words
    for doc, sentiment in zip(nlp.pipe(dataf['text']), dataf['sentiment']):
        for chunk in doc.noun_chunks:
            tokens = [token for token in chunk]
            while len(tokens) > 0 and (tokens[0].is_stop or tokens[0].pos_ == 'DET'):
                tokens.pop(0)
            while len(tokens) > 0 and tokens[-1].is_stop:
                tokens.pop(-1)
            if not tokens: continue
            final_aspect = " ".join(token.lemma_.lower() for token in tokens)
            if len(final_aspect) > 3 and final_aspect not in stop_words:
                aspect_sentiments.append({'aspect': final_aspect, 'sentiment': sentiment})
    if not aspect_sentiments:
        return pd.DataFrame()
    return pd.DataFrame(aspect_sentiments)

# --- Main App Logic ---
def main():
    st.title("⚖️ Product Comparison")

    # --- Check for Initial Product ---
    if 'selected_product' not in st.session_state or st.session_state.selected_product is None:
        st.warning("Please select a product from the main search page first to begin a comparison.")
        st.stop()
    
    product_a_asin = st.session_state.selected_product
    product_a_details = get_product_details(conn, product_a_asin).iloc[0]

    # --- Product B Selection UI ---
    st.sidebar.header("Select Product to Compare")
    all_products_df = get_filtered_products(conn, product_a_details['category'], "", "Popularity (Most Reviews)", None, None, 1000, 0)[0]
    
    # Exclude the already selected product from the list
    product_b_options = all_products_df[all_products_df['parent_asin'] != product_a_asin]
    
    selected_product_b_title = st.sidebar.selectbox(
        f"Select a product from the '{product_a_details['category']}' category to compare:",
        options=product_b_options['product_title'].tolist(),
        index=0
    )

    product_b_asin = product_b_options[product_b_options['product_title'] == selected_product_b_title]['parent_asin'].iloc[0]
    product_b_details = get_product_details(conn, product_b_asin).iloc[0]

    # --- Load Data for Both Products ---
    product_a_reviews = get_reviews_for_product(conn, product_a_asin, None, (), (), "All")
    product_b_reviews = get_reviews_for_product(conn, product_b_asin, None, (), (), "All")

    # --- Display Selected Products ---
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Product A")
        st.write(f"**{product_a_details['product_title']}**")
        st.metric("Total Reviews", f"{len(product_a_reviews):,}")
    with col2:
        st.subheader("Product B")
        st.write(f"**{product_b_details['product_title']}**")
        st.metric("Total Reviews", f"{len(product_b_reviews):,}")

    # --- 3.1 Macro-Level Sentiment: Diverging Stacked Bar Chart ---
    st.markdown("---")
    st.markdown("### Overall Sentiment Comparison")
    
    def get_sentiment_dist(df):
        dist = df['sentiment'].value_counts(normalize=True).reindex(['Positive', 'Neutral', 'Negative']).fillna(0)
        return dist

    dist_a = get_sentiment_dist(product_a_reviews)
    dist_b = get_sentiment_dist(product_b_reviews)
    
    fig = go.Figure()
    # Product A
    fig.add_trace(go.Bar(y=['Product A'], x=[dist_a['Positive']], name='Positive', orientation='h', marker_color='#1a9850'))
    fig.add_trace(go.Bar(y=['Product A'], x=[dist_a['Neutral']], name='Neutral', orientation='h', marker_color='#cccccc'))
    fig.add_trace(go.Bar(y=['Product A'], x=[-dist_a['Negative']], name='Negative', orientation='h', marker_color='#d73027'))
    # Product B
    fig.add_trace(go.Bar(y=['Product B'], x=[dist_b['Positive']], name='Positive', orientation='h', showlegend=False, marker_color='#1a9850'))
    fig.add_trace(go.Bar(y=['Product B'], x=[dist_b['Neutral']], name='Neutral', orientation='h', showlegend=False, marker_color='#cccccc'))
    fig.add_trace(go.Bar(y=['Product B'], x=[-dist_b['Negative']], name='Negative', orientation='h', showlegend=False, marker_color='#d73027'))

    fig.update_layout(
        barmode='relative', 
        xaxis_title="Proportion of Reviews", 
        yaxis_title=None,
        xaxis=dict(tickformat='.0%'),
        height=250,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- 3.2 Feature-Level Performance: Comparative Radar Chart ---
    st.markdown("---")
    st.markdown("### Feature-Level Performance Comparison")

    aspects_a = extract_aspects_with_sentiment(product_a_reviews)
    aspects_b = extract_aspects_with_sentiment(product_b_reviews)

    # Find common aspects
    common_aspects = set(aspects_a['aspect']).intersection(set(aspects_b['aspect']))
    
    if common_aspects and len(common_aspects) >= 3:
        def get_avg_sentiment_per_aspect(df, aspects_to_include):
            df_filtered = df[df['aspect'].isin(aspects_to_include)]
            sentiment_scores = df_filtered.merge(product_a_reviews[['review_id', 'sentiment_score']], on='review_id', how='left')
            return sentiment_scores.groupby('aspect')['sentiment_score'].mean()

        avg_sent_a = aspects_a.groupby('aspect')['sentiment'].apply(lambda s: (s == 'Positive').sum() - (s == 'Negative').sum()).reindex(common_aspects, fill_value=0)
        avg_sent_b = aspects_b.groupby('aspect')['sentiment'].apply(lambda s: (s == 'Positive').sum() - (s == 'Negative').sum()).reindex(common_aspects, fill_value=0)

        radar_fig = go.Figure()
        radar_fig.add_trace(go.Scatterpolar(r=avg_sent_a.values, theta=avg_sent_a.index, fill='toself', name='Product A'))
        radar_fig.add_trace(go.Scatterpolar(r=avg_sent_b.values, theta=avg_sent_b.index, fill='toself', name='Product B'))
        st.plotly_chart(radar_fig, use_container_width=True)
    else:
        st.info("Not enough common aspects found between these two products to generate a comparison chart.")

    # --- 3.3 Qualitative Insights: Juxtaposed Aspect-Driven Word Clouds ---
    st.markdown("---")
    st.markdown("### Qualitative Keyword Comparison")
    
    if common_aspects:
        wc_col1, wc_col2 = st.columns(2)
        with wc_col1:
            selected_aspect_wc = st.selectbox("Select an aspect to compare keywords:", options=list(common_aspects))
        with wc_col2:
            selected_sentiment_wc = st.selectbox("Select a sentiment:", options=['Positive', 'Negative'])

        def generate_wordcloud(df, aspect, sentiment):
            text_corpus = df[(df['aspect'] == aspect) & (df['sentiment'] == sentiment)]['text']
            if text_corpus.empty:
                return None
            return WordCloud(stopwords='english', background_color="white", colormap='Greens' if sentiment=='Positive' else 'Reds').generate(" ".join(text_corpus))
        
        # Combine aspect and review text for word cloud generation
        aspect_text_a = aspects_a.merge(product_a_reviews[['review_id', 'text']], on='review_id')
        aspect_text_b = aspects_b.merge(product_b_reviews[['review_id', 'text']], on='review_id')

        wc_a = generate_wordcloud(aspect_text_a, selected_aspect_wc, selected_sentiment_wc)
        wc_b = generate_wordcloud(aspect_text_b, selected_aspect_wc, selected_sentiment_wc)

        wc_disp_col1, wc_disp_col2 = st.columns(2)
        with wc_disp_col1:
            st.write("**Product A Keywords**")
            if wc_a:
                fig, ax = plt.subplots()
                ax.imshow(wc_a, interpolation='bilinear')
                ax.axis("off")
                st.pyplot(fig)
            else:
                st.caption(f"No {selected_sentiment_wc.lower()} reviews found for this aspect.")
        
        with wc_disp_col2:
            st.write("**Product B Keywords**")
            if wc_b:
                fig, ax = plt.subplots()
                ax.imshow(wc_b, interpolation='bilinear')
                ax.axis("off")
                st.pyplot(fig)
            else:
                st.caption(f"No {selected_sentiment_wc.lower()} reviews found for this aspect.")
    else:
        st.info("No common aspects found for keyword comparison.")

if __name__ == "__main__":
    main()
