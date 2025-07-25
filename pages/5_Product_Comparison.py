# pages/5_Product_Comparison.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import spacy
import re
import json
import altair as alt
from utils.database_utils import (
    connect_to_db,
    get_product_details,
    get_reviews_for_product,
    get_all_categories,
    get_filtered_products,
    get_product_date_range
)

# --- Page Config & Model Loading ---
st.set_page_config(layout="wide", page_title="Product Comparison")

@st.cache_resource
def load_spacy_model():
    return spacy.load("en_core_web_sm")

nlp = load_spacy_model()
DB_PATH = "amazon_reviews_final.duckdb"
conn = connect_to_db(DB_PATH)

# --- Definitive Aspect Extraction Function ---
@st.cache_data
def extract_aspects_with_sentiment(dataf):
    aspect_sentiments = []
    stop_words = nlp.Defaults.stop_words
    
    # --- FIX: Pass the review_id into the zip function ---
    for doc, sentiment, review_id in zip(nlp.pipe(dataf['text']), dataf['sentiment'], dataf['review_id']):
        for chunk in doc.noun_chunks:
            tokens = [token for token in chunk]
            while len(tokens) > 0 and (tokens[0].is_stop or tokens[0].pos_ == 'DET'):
                tokens.pop(0)
            while len(tokens) > 0 and tokens[-1].is_stop:
                tokens.pop(-1)
            if not tokens: continue
            
            final_aspect = " ".join(token.lemma_.lower() for token in tokens)
            
            if len(final_aspect) > 3 and final_aspect not in stop_words:
                # --- FIX: Add the review_id to the dictionary ---
                aspect_sentiments.append({
                    'aspect': final_aspect, 
                    'sentiment': sentiment,
                    'review_id': review_id 
                })

    if not aspect_sentiments:
        return pd.DataFrame()
        
    return pd.DataFrame(aspect_sentiments)

# --- Main App Logic ---
def main():
    st.title("⚖️ Product Comparison")

    if 'selected_product' not in st.session_state or st.session_state.selected_product is None:
        st.warning("Please select a product from the main search page first to begin a comparison.")
        st.stop()
    
    product_a_asin = st.session_state.selected_product
    product_a_details = get_product_details(conn, product_a_asin).iloc[0]

    # --- Product B Selection ---
    st.sidebar.header("Select Product to Compare")
    all_products_df = get_filtered_products(conn, product_a_details['category'], "", "Popularity (Most Reviews)", None, None, 1000, 0)[0]
    product_b_options = all_products_df[all_products_df['parent_asin'] != product_a_asin]
    selected_product_b_title = st.sidebar.selectbox(
        f"Select a product from the '{product_a_details['category']}' category:",
        options=product_b_options['product_title'].tolist(),
        index=0, key="product_b_selector"
    )
    product_b_asin = product_b_options[product_b_options['product_title'] == selected_product_b_title]['parent_asin'].iloc[0]
    product_b_details = get_product_details(conn, product_b_asin).iloc[0]

    # --- Consistent Sidebar Filters ---
    st.sidebar.header("🔬 Comparison Filters")
    min_date_a, max_date_a = get_product_date_range(conn, product_a_asin)
    min_date_b, max_date_b = get_product_date_range(conn, product_b_asin)
    
    selected_date_range = st.sidebar.date_input("Filter by Date Range", value=(min(min_date_a, min_date_b), max(max_date_a, max_date_b)), key='compare_date_filter')
    selected_ratings = st.sidebar.multiselect("Filter by Star Rating", options=[1, 2, 3, 4, 5], default=[1, 2, 3, 4, 5], key='compare_rating_filter')
    selected_sentiments = st.sidebar.multiselect("Filter by Sentiment", options=['Positive', 'Negative', 'Neutral'], default=['Positive', 'Negative', 'Neutral'], key='compare_sentiment_filter')
    selected_verified = st.sidebar.radio("Filter by Purchase Status", ["All", "Verified Only", "Not Verified"], index=0, key='compare_verified_filter')

    # --- Load Filtered Data for Both Products ---
    product_a_reviews = get_reviews_for_product(conn, product_a_asin, selected_date_range, tuple(selected_ratings), tuple(selected_sentiments), selected_verified)
    product_b_reviews = get_reviews_for_product(conn, product_b_asin, selected_date_range, tuple(selected_ratings), tuple(selected_sentiments), selected_verified)

    # --- Display Panes ---
    col1, col2 = st.columns(2)
    with col1:
        st.subheader(product_a_details['product_title'])
        st.metric("Filtered Reviews", f"{len(product_a_reviews):,}")
    with col2:
        st.subheader(product_b_details['product_title'])
        st.metric("Filtered Reviews", f"{len(product_b_reviews):,}")
        
    st.markdown("---")
    st.markdown("### Overall Sentiment and Rating Comparison")
    st.info("These charts directly compare the proportion of sentiments and star ratings for each product. Hover over the bars to see the raw counts.")

    # --- Helper function to truncate long text ---
    def truncate_text(text, max_length=10):
        return text if len(text) <= max_length else text[:max_length] + "..."

    # Get the (potentially truncated) product titles for the legend
    product_a_title = truncate_text(product_a_details['product_title'])
    product_b_title = truncate_text(product_b_details['product_title'])

    # --- Create a two-column layout for the charts ---
    col1, col2 = st.columns(2)

    # --- Column 1: Grouped Sentiment Bar Chart ---
    with col1:
        st.markdown("**Sentiment Distribution**")
        
        counts_a = product_a_reviews['sentiment'].value_counts().reindex(['Positive', 'Neutral', 'Negative']).fillna(0)
        dist_a = counts_a / counts_a.sum() if counts_a.sum() > 0 else counts_a
        counts_b = product_b_reviews['sentiment'].value_counts().reindex(['Positive', 'Neutral', 'Negative']).fillna(0)
        dist_b = counts_b / counts_b.sum() if counts_b.sum() > 0 else counts_b
        
        df_a = pd.DataFrame({'Proportion': dist_a, 'Count': counts_a}).reset_index(); df_a.columns = ['Sentiment', 'Proportion', 'Count']; df_a['Product'] = product_a_title
        df_b = pd.DataFrame({'Proportion': dist_b, 'Count': counts_b}).reset_index(); df_b.columns = ['Sentiment', 'Proportion', 'Count']; df_b['Product'] = product_b_title
        plot_df = pd.concat([df_a, df_b])

        sentiment_chart = alt.Chart(plot_df).mark_bar().encode(
            x=alt.X('Sentiment:N', title="Sentiment", sort=['Positive', 'Neutral', 'Negative']),
            y=alt.Y('Proportion:Q', title="Proportion of Reviews", axis=alt.Axis(format='%')),
            color=alt.Color('Product:N', scale=alt.Scale(range=['#4c78a8', '#f58518'])),
            xOffset='Product:N',
            tooltip=[
                alt.Tooltip('Product:N'), alt.Tooltip('Sentiment:N'),
                alt.Tooltip('Count:Q', title='Review Count'),
                alt.Tooltip('Proportion:Q', title='Proportion', format='.1%')
            ]
        ).properties(title="Sentiment Comparison")
        st.altair_chart(sentiment_chart, use_container_width=True)

    # --- Column 2: Grouped Rating Bar Chart ---
    with col2:
        st.markdown("**Rating Distribution**")

        rating_counts_a = product_a_reviews['rating'].value_counts().reindex([5, 4, 3, 2, 1]).fillna(0)
        rating_dist_a = rating_counts_a / rating_counts_a.sum() if rating_counts_a.sum() > 0 else rating_counts_a
        rating_counts_b = product_b_reviews['rating'].value_counts().reindex([5, 4, 3, 2, 1]).fillna(0)
        rating_dist_b = rating_counts_b / rating_counts_b.sum() if rating_counts_b.sum() > 0 else rating_counts_b
        
        df_a_ratings = pd.DataFrame({'Proportion': rating_dist_a, 'Count': rating_counts_a}).reset_index(); df_a_ratings.columns = ['Rating', 'Proportion', 'Count']; df_a_ratings['Product'] = product_a_title
        df_b_ratings = pd.DataFrame({'Proportion': rating_dist_b, 'Count': rating_counts_b}).reset_index(); df_b_ratings.columns = ['Rating', 'Proportion', 'Count']; df_b_ratings['Product'] = product_b_title
        plot_df_ratings = pd.concat([df_a_ratings, df_b_ratings])

        rating_chart = alt.Chart(plot_df_ratings).mark_bar().encode(
            x=alt.X('Rating:O', title="Star Rating", sort=alt.EncodingSortField(field="Rating", order="descending")),
            y=alt.Y('Proportion:Q', title="Proportion of Reviews", axis=alt.Axis(format='%')),
            color=alt.Color('Product:N', scale=alt.Scale(range=['#4c78a8', '#f58518'])),
            xOffset='Product:N',
            tooltip=[
                alt.Tooltip('Product:N'), alt.Tooltip('Rating:O'),
                alt.Tooltip('Count:Q', title='Review Count'),
                alt.Tooltip('Proportion:Q', title='Proportion', format='.1%')
            ]
        ).properties(title="Rating Comparison")
        st.altair_chart(rating_chart, use_container_width=True)
        
    # --- Feature-Level Performance: Comparative Radar Charts ---
# --- Feature-Level Performance: Comparative Radar Chart ---
    st.markdown("---")
    st.markdown("### Feature-Level Performance Comparison")
    st.info(
        "This radar chart compares the average sentiment score for the top common aspects. "
        "Each colored line represents an aspect, showing its score for Product A vs. Product B."
    )

    aspects_a = extract_aspects_with_sentiment(product_a_reviews)
    aspects_b = extract_aspects_with_sentiment(product_b_reviews)
    
    if not aspects_a.empty and not aspects_b.empty:
        counts_a = aspects_a['aspect'].value_counts()
        counts_b = aspects_b['aspect'].value_counts()
        common_aspects = set(counts_a.index).intersection(set(counts_b.index))
        
        if len(common_aspects) >= 3:
            total_counts = (counts_a.reindex(common_aspects, fill_value=0) + counts_b.reindex(common_aspects, fill_value=0)).sort_values(ascending=False)
            
            # --- FIX: Set max_value to 10 ---
            num_aspects_to_show = st.slider(
                "Select number of top aspects to display:",
                min_value=3, 
                max_value=min(10, len(total_counts)), # Max is now 10
                value=min(7, len(total_counts)),
                key="radar_aspect_slider"
            )
            
            top_common_aspects = total_counts.nlargest(num_aspects_to_show).index.tolist()

            aspects_a = aspects_a.merge(product_a_reviews[['review_id', 'sentiment_score']], on='review_id')
            aspects_b = aspects_b.merge(product_b_reviews[['review_id', 'sentiment_score']], on='review_id')
            
            avg_sent_a = aspects_a.groupby('aspect')['sentiment_score'].mean().reindex(top_common_aspects)
            avg_sent_b = aspects_b.groupby('aspect')['sentiment_score'].mean().reindex(top_common_aspects)

            # --- FIX: Restructure data and chart for aspect-based legend ---
            product_a_title = truncate_text(product_a_details['product_title'])
            product_b_title = truncate_text(product_b_details['product_title'])
            
            # The axes of our new chart will be the two products
            theta_labels = [product_a_title, product_b_title]

            fig = go.Figure()

            # Add a separate trace for each aspect
            for aspect in top_common_aspects:
                fig.add_trace(go.Scatterpolar(
                    r=[avg_sent_a[aspect], avg_sent_b[aspect]], # Scores for Product A and B
                    theta=theta_labels, 
                    fill='toself',
                    name=aspect.title() # Use the aspect name in the legend
                ))
            
            fig.update_layout(
              polar=dict(radialaxis=dict(visible=True, range=[-1, 1])),
              showlegend=True,
              title="Comparative Sentiment Scores by Aspect"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough common aspects (at least 3) were found to generate a comparison chart.")
    else:
        st.info("Not enough aspect data for one or both products to generate a comparison.")
if __name__ == "__main__":
    main()
