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

# --- Helper Functions ---
def get_rating_consensus(std_dev):
    """Helper function to interpret standard deviation."""
    if std_dev < 1.1:
        return "✅ Consistent"
    elif std_dev < 1.4:
        return "↔️ Mixed"
    else:
        return "⚠️ Polarizing"
        
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
        # --- Add the back button here ---
    if st.button("⬅️ Back to Sentiment Overview"):
        st.switch_page("pages/1_Sentiment_Overview.py")
 
    if 'product_b_asin' not in st.session_state:
        st.session_state.product_b_asin = None
        
    # In main(), replace the section from "Load Product A" down to "Load Product B"
    # --- 1. Load Product A (from session state) ---
    if 'selected_product' not in st.session_state or st.session_state.selected_product is None:
        st.warning("Please select a product from the main search page first.")
        st.stop()
    product_a_asin = st.session_state.selected_product
    product_a_details = get_product_details(conn, product_a_asin).iloc[0]

    # --- 2. Create the main layout ---
    col_a, col_b = st.columns(2, gap="large")

    # --- COLUMN A: Display the primary product ---
    with col_a:
        st.subheader(product_a_details['product_title'])
        image_url_a = (product_a_details.get('image_urls') or '').split(',')[0]
        if image_url_a:
            st.image(image_url_a, use_container_width=True)
        st.info("This is the product you are comparing against.")
        
    # --- COLUMN B: Handles competitor selection and display ---
    with col_b:
        # If no competitor has been chosen yet, show the search UI
        if st.session_state.product_b_asin is None:
            st.subheader("Select a Product to Compare")

            # Search and filter controls for finding Product B
            search_term_b = st.text_input("Search for a competitor in the same category:")
            
            # Callback to set the selected product
            def select_product_b(asin):
                st.session_state.product_b_asin = asin

            # Fetch and display potential competitors
            competitors_df, _ = get_filtered_products(
                conn,
                product_a_details['category'],
                search_term_b,
                "Popularity (Most Reviews)",
                None, None, 10, 0
            )
            competitors_df = competitors_df[competitors_df['parent_asin'] != product_a_asin]

            if competitors_df.empty:
                st.warning("No competitors found with that search term.")
            else:
                for _, row in competitors_df.iterrows():
                    with st.container(border=True):
                        c_img, c_details = st.columns([0.3, 0.7])
                        with c_img:
                            img_url = (row.get('image_urls') or '').split(',')[0]
                            if img_url: st.image(img_url)
                        with c_details:
                            st.markdown(f"**{row['product_title']}**")
                            st.caption(f"Rating: {row.get('average_rating', 0):.2f} ⭐ | Reviews: {row.get('review_count', 0)}")
                            st.button("Select to Compare", key=f"select_{row['parent_asin']}", on_click=select_product_b, args=(row['parent_asin'],))
        
        # If a competitor HAS been chosen, we will eventually display its details here
        else:
            # This 'else' block is where Product B's "At a Glance" metrics will go later.
            # For now, we can add a placeholder and a button to reset the selection.
            product_b_details = get_product_details(conn, st.session_state.product_b_asin).iloc[0]
            st.subheader(product_b_details['product_title'])
            image_url_b = (product_b_details.get('image_urls') or '').split(',')[0]
            if image_url_b:
                st.image(image_url_b, use_container_width=True)

            if st.button("Change Competitor", use_container_width=True):
                st.session_state.product_b_asin = None
                st.rerun()
    if st.session_state.product_b_asin:
        # Use a shared set of filters for a fair comparison
        st.sidebar.header("🔬 Comparison Filters")
        min_date_a, max_date_a = get_product_date_range(conn, product_a_asin)
        min_date_b, max_date_b = get_product_date_range(conn, product_b_asin)
        selected_date_range = st.sidebar.date_input("Filter by Date Range", value=(min(min_date_a, min_date_b), max(max_date_a, max_date_b)), key='compare_date_filter')
        selected_ratings = st.sidebar.multiselect("Filter by Star Rating", options=[1, 2, 3, 4, 5], default=[1, 2, 3, 4, 5], key='compare_rating_filter')
    
        # Load review data for both products using the shared filters
        product_a_reviews = get_reviews_for_product(conn, product_a_asin, selected_date_range, tuple(selected_ratings), (), "All")
        product_b_reviews = get_reviews_for_product(conn, product_b_asin, selected_date_range, tuple(selected_ratings), (), "All")
    
        # =================================================================
        # SECTION 1: AT A GLANCE
        # =================================================================
        st.markdown("---")
        st.header("⭐ At a Glance")
    
        # --- Data Calculation ---
        # Product A
        avg_rating_a = product_a_details.get('average_rating', 0)
        avg_sentiment_a = product_a_reviews['sentiment_score'].mean() if not product_a_reviews.empty else 0
        consensus_a = get_rating_consensus(product_a_reviews['rating'].std()) if not product_a_reviews.empty and len(product_a_reviews) > 1 else "N/A"
        verified_a = (product_a_reviews['verified_purchase'].sum() / len(product_a_reviews)) * 100 if not product_a_reviews.empty else 0
    
        # Product B
        avg_rating_b = product_b_details.get('average_rating', 0)
        avg_sentiment_b = product_b_reviews['sentiment_score'].mean() if not product_b_reviews.empty else 0
        consensus_b = get_rating_consensus(product_b_reviews['rating'].std()) if not product_b_reviews.empty and len(product_b_reviews) > 1 else "N/A"
        verified_b = (product_b_reviews['verified_purchase'].sum() / len(product_b_reviews)) * 100 if not product_b_reviews.empty else 0
    
            # --- Display Layout ---
        col1, col2 = st.columns(2)
    
        # --- Column 1: Product A ---
        with col1:
            st.subheader(product_a_details['product_title'])
            image_url_a = (product_a_details.get('image_urls') or '').split(',')[0]
            if image_url_a: st.image(image_url_a, use_container_width=True)
    
            # 2x2 grid for metrics
            row1_c1, row1_c2 = st.columns(2)
            with row1_c1:
                st.metric("Average Rating", f"{avg_rating_a:.2f} ⭐", delta=f"{avg_rating_a - avg_rating_b:.2f}")
            with row1_c2:
                st.metric("Avg. Sentiment", f"{avg_sentiment_a:.2f}", delta=f"{avg_sentiment_a - avg_sentiment_b:.2f}")
    
            row2_c1, row2_c2 = st.columns(2)
            with row2_c1:
                st.metric("Reviewer Consensus", consensus_a)
            with row2_c2:
                st.metric("Verified Purchases", f"{verified_a:.1f}%", delta=f"{verified_a - verified_b:.1f}%")
    
        # --- Column 2: Product B ---
        with col2:
            st.subheader(product_b_details['product_title'])
            image_url_b = (product_b_details.get('image_urls') or '').split(',')[0]
            if image_url_b: st.image(image_url_b, use_container_width=True)
    
            # 2x2 grid for metrics
            row1_c1, row1_c2 = st.columns(2)
            with row1_c1:
                st.metric("Average Rating", f"{avg_rating_b:.2f} ⭐", delta=f"{avg_rating_b - avg_rating_a:.2f}")
            with row1_c2:
                st.metric("Avg. Sentiment", f"{avg_sentiment_b:.2f}", delta=f"{avg_sentiment_b - avg_sentiment_a:.2f}")
    
            row2_c1, row2_c2 = st.columns(2)
            with row2_c1:
                st.metric("Reviewer Consensus", consensus_b)
            with row2_c2:
                st.metric("Verified Purchases", f"{verified_b:.1f}%", delta=f"{verified_b - verified_a:.1f}%")
                
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
            
        # In pages/5_Product_Comparison.py
    
        # --- Feature-Level Performance: Comparative Radar Chart ---
        st.markdown("---")
        st.markdown("### Feature-Level Performance Comparison")
        st.info("This radar chart directly compares the average sentiment score for the most frequently discussed common aspects.")
    
        aspects_a = extract_aspects_with_sentiment(product_a_reviews)
        aspects_b = extract_aspects_with_sentiment(product_b_reviews)
        
        if not aspects_a.empty and not aspects_b.empty:
            counts_a = aspects_a['aspect'].value_counts()
            counts_b = aspects_b['aspect'].value_counts()
            common_aspects = set(counts_a.index).intersection(set(counts_b.index))
            
            if len(common_aspects) >= 3:
                total_counts = (counts_a.reindex(common_aspects, fill_value=0) + counts_b.reindex(common_aspects, fill_value=0)).sort_values(ascending=False)
                
                num_aspects_to_show = st.slider(
                    "Select number of top aspects to display:",
                    min_value=3, 
                    max_value=min(20, len(total_counts)),
                    value=min(5, len(total_counts)),
                    key="radar_aspect_slider"
                )
                
                top_common_aspects = total_counts.nlargest(num_aspects_to_show).index.tolist()
    
                aspects_a = aspects_a.merge(product_a_reviews[['review_id', 'sentiment_score']], on='review_id')
                aspects_b = aspects_b.merge(product_b_reviews[['review_id', 'sentiment_score']], on='review_id')
                
                avg_sent_a = aspects_a.groupby('aspect')['sentiment_score'].mean().reindex(top_common_aspects)
                avg_sent_b = aspects_b.groupby('aspect')['sentiment_score'].mean().reindex(top_common_aspects)
    
                product_a_title = truncate_text(product_a_details['product_title'])
                product_b_title = truncate_text(product_b_details['product_title'])
    
                # --- FIX: Create a single, combined radar chart ---
                fig = go.Figure()
    
                fig.add_trace(go.Scatterpolar(
                    r=avg_sent_a.values,
                    theta=avg_sent_a.index, 
                    fill='toself',
                    name=product_a_title,
                    marker_color='#4c78a8', # Professional Blue
                    opacity=0.7
                ))
                fig.add_trace(go.Scatterpolar(
                    r=avg_sent_b.values,
                    theta=avg_sent_b.index,
                    fill='toself',
                    name=product_b_title,
                    marker_color='#f58518', # Professional Orange
                    opacity=0.7
                ))
                
                fig.update_layout(
                  polar=dict(radialaxis=dict(visible=True, range=[-1, 1])),
                  showlegend=True,
                  title="Comparative Sentiment Scores by Aspect"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Not enough common aspects (at least 3) found to generate a comparison chart.")
        else:
            st.info("Not enough aspect data for one or both products to generate a comparison.")
if __name__ == "__main__":
    main()
