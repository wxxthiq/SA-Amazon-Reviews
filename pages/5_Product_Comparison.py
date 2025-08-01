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
    """Loads the spaCy model."""
    return spacy.load("en_core_web_sm")

nlp = load_spacy_model()
DB_PATH = "amazon_reviews_final.duckdb"
conn = connect_to_db(DB_PATH)

# --- Definitive Aspect Extraction Function ---
@st.cache_data
def extract_aspects_with_sentiment(dataf):
    """Extracts aspects and their associated sentiment from review text."""
    aspect_sentiments = []
    stop_words = nlp.Defaults.stop_words

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
    """Main function to run the Streamlit page."""
    st.title("⚖️ Product Comparison")

    if st.button("⬅️ Back to Sentiment Overview"):
        st.switch_page("pages/1_Sentiment_Overview.py")

    if 'selected_product' not in st.session_state or st.session_state.selected_product is None:
        st.warning("Please select a product from the main search page first to begin a comparison.")
        st.stop()

    product_a_asin = st.session_state.selected_product
    product_a_details = get_product_details(conn, product_a_asin).iloc[0]

    # --- Product B Selection ---
    st.sidebar.header("Select Product to Compare")
    all_products_df = get_filtered_products(conn, product_a_details['category'], "", "Popularity (Most Reviews)", None, None, 1000, 0)[0]
    product_b_options = all_products_df[all_products_df['parent_asin'] != product_a_asin]

    st.subheader("Select a Product for Comparison")
    st.write(f"You are currently comparing products from the '{product_a_details['category']}' category.")

    # Display product cards for selection
    cols = st.columns(4)
    for i, col in enumerate(cols):
        if i < len(product_b_options):
            product = product_b_options.iloc[i]
            with col:
                with st.container(border=True):
                    image_urls_str = product.get('image_urls')
                    thumbnail_url = image_urls_str.split(',')[0] if pd.notna(image_urls_str) else "https://via.placeholder.com/200"
                    st.image(thumbnail_url, use_container_width=True)
                    st.markdown(f"**{product['product_title']}**")
                    if st.button("Select", key=product['parent_asin']):
                        st.session_state.product_b_asin = product['parent_asin']

    if 'product_b_asin' not in st.session_state:
        st.info("Please select a product from the cards above to start the comparison.")
        st.stop()

    product_b_asin = st.session_state.product_b_asin
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

    def get_rating_consensus(std_dev):
        """Helper function to interpret standard deviation."""
        if std_dev is None or pd.isna(std_dev):
            return "N/A"
        if std_dev < 1.1:
            return "✅ Consistent"
        elif std_dev < 1.4:
            return "↔️ Mixed"
        else:
            return "⚠️ Polarizing"

    st.markdown("---")
    st.subheader("Product Overview")

    col1, col2 = st.columns(2)

    # --- Product A Display ---
    with col1:
        st.subheader(product_a_details['product_title'])
        image_urls_str_a = product_a_details.get('image_urls')
        image_urls_a = image_urls_str_a.split(',') if pd.notna(image_urls_str_a) and image_urls_str_a else []
        if image_urls_a:
            st.image(image_urls_a[0], use_container_width=True)

        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("Average Rating", f"{product_a_details.get('average_rating', 0):.2f} ⭐")
        m_col2.metric("Filtered Reviews", f"{len(product_a_reviews):,}")

        if not product_a_reviews.empty and len(product_a_reviews) > 1:
            std_dev_a = product_a_reviews['rating'].std()
            consensus_a = get_rating_consensus(std_dev_a)
            m_col3.metric("Consensus", consensus_a, help=f"Std. Dev: {std_dev_a:.2f}")

    # --- Product B Display ---
    with col2:
        st.subheader(product_b_details['product_title'])
        image_urls_str_b = product_b_details.get('image_urls')
        image_urls_b = image_urls_str_b.split(',') if pd.notna(image_urls_str_b) and image_urls_str_b else []
        if image_urls_b:
            st.image(image_urls_b[0], use_container_width=True)
            
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("Average Rating", f"{product_b_details.get('average_rating', 0):.2f} ⭐")
        m_col2.metric("Filtered Reviews", f"{len(product_b_reviews):,}")

        if not product_b_reviews.empty and len(product_b_reviews) > 1:
            std_dev_b = product_b_reviews['rating'].std()
            consensus_b = get_rating_consensus(std_dev_b)
            m_col3.metric("Consensus", consensus_b, help=f"Std. Dev: {std_dev_b:.2f}")
        
    st.markdown("---")
    st.markdown("### Overall Sentiment and Rating Comparison")
    st.info("These charts directly compare the proportion of sentiments and star ratings for each product. Hover over the bars to see the raw counts.")

    def truncate_text(text, max_length=25):
        """Helper function to truncate long text."""
        return text if len(text) <= max_length else text[:max_length] + "..."

    product_a_title = truncate_text(product_a_details['product_title'])
    product_b_title = truncate_text(product_b_details['product_title'])

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Sentiment Comparison")
        counts_a = product_a_reviews['sentiment'].value_counts().reindex(['Positive', 'Neutral', 'Negative']).fillna(0)
        counts_b = product_b_reviews['sentiment'].value_counts().reindex(['Positive', 'Neutral', 'Negative']).fillna(0)
        
        df_a = pd.DataFrame({'Count': counts_a}).reset_index(); df_a.columns = ['Sentiment', 'Count']; df_a['Product'] = product_a_title
        df_b = pd.DataFrame({'Count': counts_b}).reset_index(); df_b.columns = ['Sentiment', 'Count']; df_b['Product'] = product_b_title
        plot_df = pd.concat([df_a, df_b])

        sentiment_chart = alt.Chart(plot_df).mark_bar().encode(
            y=alt.Y('Product:N', title=None),
            x=alt.X('sum(Count):Q', stack="normalize", title="Proportion of Reviews", axis=alt.Axis(format='%')),
            color=alt.Color('Sentiment:N', scale=alt.Scale(domain=['Positive', 'Neutral', 'Negative'], range=['#1a9850', '#cccccc', '#d73027']), legend=alt.Legend(title="Sentiment")),
            tooltip=[alt.Tooltip('Product:N'), alt.Tooltip('Sentiment:N'), alt.Tooltip('sum(Count):Q', title='Review Count')]
        )
        st.altair_chart(sentiment_chart, use_container_width=True)

    with col2:
        st.markdown("#### Rating Comparison")
        rating_counts_a = product_a_reviews['rating'].value_counts().reindex([5, 4, 3, 2, 1]).fillna(0)
        rating_counts_b = product_b_reviews['rating'].value_counts().reindex([5, 4, 3, 2, 1]).fillna(0)
        
        df_a_ratings = pd.DataFrame({'Count': rating_counts_a}).reset_index(); df_a_ratings.columns = ['Rating', 'Count']; df_a_ratings['Product'] = product_a_title
        df_b_ratings = pd.DataFrame({'Count': rating_counts_b}).reset_index(); df_b_ratings.columns = ['Rating', 'Count']; df_b_ratings['Product'] = product_b_title
        plot_df_ratings = pd.concat([df_a_ratings, df_b_ratings])

        rating_chart = alt.Chart(plot_df_ratings).mark_bar().encode(
            y=alt.Y('Product:N', title=None),
            x=alt.X('sum(Count):Q', stack="normalize", title="Proportion of Reviews", axis=alt.Axis(format='%')),
            color=alt.Color('Rating:O', scale=alt.Scale(domain=[5, 4, 3, 2, 1], range=['#2ca02c', '#98df8a', '#ffdd71', '#ff9896', '#d62728']), legend=alt.Legend(title="Rating")),
            tooltip=[alt.Tooltip('Product:N'), alt.Tooltip('Rating:O'), alt.Tooltip('sum(Count):Q', title='Review Count')]
        )
        st.altair_chart(rating_chart, use_container_width=True)
        
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

            product_a_title = truncate_text(product_a_details['product_title'], max_length=15)
            product_b_title = truncate_text(product_b_details['product_title'], max_length=15)

            fig = go.Figure()

            fig.add_trace(go.Scatterpolar(
                r=avg_sent_a.values,
                theta=avg_sent_a.index, 
                fill='toself',
                name=product_a_title,
                marker_color='#4c78a8',
                opacity=0.7
            ))
            fig.add_trace(go.Scatterpolar(
                r=avg_sent_b.values,
                theta=avg_sent_b.index,
                fill='toself',
                name=product_b_title,
                marker_color='#f58518',
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
