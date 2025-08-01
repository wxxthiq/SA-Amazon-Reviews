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
PLACEHOLDER_IMAGE_URL = "https://via.placeholder.com/200"


# --- Definitive Aspect Extraction Function ---
@st.cache_data
def extract_aspects_with_sentiment(dataf):
    """Extracts aspects and their associated sentiment from review text."""
    aspect_sentiments = []
    stop_words = nlp.Defaults.stop_words

    # Pass the review_id into the zip function
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
                # Add the review_id to the dictionary
                aspect_sentiments.append({
                    'aspect': final_aspect,
                    'sentiment': sentiment,
                    'review_id': review_id
                })

    if not aspect_sentiments:
        return pd.DataFrame()

    return pd.DataFrame(aspect_sentiments)

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

# --- Main App Logic ---
def main():
    st.title("⚖️ Product Comparison")

    if st.button("⬅️ Back to Sentiment Overview"):
        st.session_state.pop('product_b_asin', None) # Clear selection on exit
        st.switch_page("pages/1_Sentiment_Overview.py")

    if 'selected_product' not in st.session_state or st.session_state.selected_product is None:
        st.warning("Please select a product from the main search page first to begin a comparison.")
        st.stop()

    # --- Data Loading for Product A ---
    product_a_asin = st.session_state.selected_product
    product_a_details = get_product_details(conn, product_a_asin).iloc[0]

    # --- Two-Column Layout ---
    col1, col2 = st.columns(2)

    # --- Left Column: Display Product A Details ---
    with col1:
        st.header("Your Selected Product")
        with st.container(border=True):
            st.subheader(product_a_details['product_title'])
            image_urls_str_a = product_a_details.get('image_urls')
            image_urls_a = image_urls_str_a.split(',') if pd.notna(image_urls_str_a) and image_urls_str_a else []
            if image_urls_a:
                st.image(image_urls_a[0], use_container_width=True)
            else:
                st.image(PLACEHOLDER_IMAGE_URL, use_container_width=True)

            st.metric("Overall Average Rating", f"{product_a_details.get('average_rating', 0):.2f} ⭐")
            st.metric("Total Reviews", f"{int(product_a_details.get('review_count', 0)):,}")
            st.caption(f"Category: {product_a_details['category']}")


    # --- Right Column: Product B Selection or Display ---
    with col2:
        # If Product B hasn't been selected yet, show the selection pane
        if 'product_b_asin' not in st.session_state:
            st.header("Select Product to Compare")
            st.info(f"Showing other products from the **'{product_a_details['category']}'** category. Click 'Select' on a product card to begin the comparison.")

            all_products_df = get_filtered_products(conn, product_a_details['category'], "", "Popularity (Most Reviews)", None, None, 1000, 0)[0]
            product_b_options = all_products_df[all_products_df['parent_asin'] != product_a_asin]

            # Display product cards for selection
            for i in range(0, len(product_b_options), 2):
                row_cols = st.columns(2)
                for j, row_col in enumerate(row_cols):
                    if i + j < len(product_b_options):
                        product = product_b_options.iloc[i+j]
                        with row_col:
                            with st.container(border=True):
                                image_urls_str = product.get('image_urls')
                                thumbnail_url = image_urls_str.split(',')[0] if pd.notna(image_urls_str) else PLACEHOLDER_IMAGE_URL
                                st.image(thumbnail_url, use_container_width=True)
                                st.markdown(f"**{product['product_title']}**")
                                if st.button("Select", key=product['parent_asin'], use_container_width=True):
                                    st.session_state.product_b_asin = product['parent_asin']
                                    st.rerun()
        else:
            # If Product B is selected, display its details
            st.header("Comparison Product")
            product_b_asin = st.session_state.product_b_asin
            product_b_details = get_product_details(conn, product_b_asin).iloc[0]

            with st.container(border=True):
                st.subheader(product_b_details['product_title'])
                image_urls_str_b = product_b_details.get('image_urls')
                image_urls_b = image_urls_str_b.split(',') if pd.notna(image_urls_str_b) and image_urls_str_b else []
                if image_urls_b:
                    st.image(image_urls_b[0], use_container_width=True)
                else:
                    st.image(PLACEHOLDER_IMAGE_URL, use_container_width=True)

                st.metric("Overall Average Rating", f"{product_b_details.get('average_rating', 0):.2f} ⭐")
                st.metric("Total Reviews", f"{int(product_b_details.get('review_count', 0)):,}")
                st.caption(f"Category: {product_b_details['category']}")
                if st.button("Change Product", use_container_width=True):
                    st.session_state.pop('product_b_asin', None)
                    st.rerun()


    # --- Side-by-Side Comparison Section (appears only after Product B is selected) ---
    if 'product_b_asin' in st.session_state:
        st.markdown("---")
        st.header("Side-by-Side Comparison")

        # --- Sidebar Filters ---
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

        # --- Display Filtered Metrics ---
        st.subheader("Filtered Review Analysis")
        st.info("The metrics and charts below are based on the filters you select in the sidebar.")
        filtered_col1, filtered_col2 = st.columns(2)
        with filtered_col1:
             with st.container(border=True):
                st.markdown(f"**{product_a_details['product_title']}**")
                m_col1, m_col2 = st.columns(2)
                m_col1.metric("Filtered Reviews", f"{len(product_a_reviews):,}")
                if not product_a_reviews.empty and len(product_a_reviews) > 1:
                    std_dev_a = product_a_reviews['rating'].std()
                    consensus_a = get_rating_consensus(std_dev_a)
                    m_col2.metric("Reviewer Consensus", consensus_a, help=f"Std. Dev: {std_dev_a:.2f}")

        with filtered_col2:
            with st.container(border=True):
                st.markdown(f"**{product_b_details['product_title']}**")
                m_col1, m_col2 = st.columns(2)
                m_col1.metric("Filtered Reviews", f"{len(product_b_reviews):,}")
                if not product_b_reviews.empty and len(product_b_reviews) > 1:
                    std_dev_b = product_b_reviews['rating'].std()
                    consensus_b = get_rating_consensus(std_dev_b)
                    m_col2.metric("Reviewer Consensus", consensus_b, help=f"Std. Dev: {std_dev_b:.2f}")

        # --- Comparison Charts ---
        def truncate_text(text, max_length=15):
            return text if len(text) <= max_length else text[:max_length] + "..."

        product_a_title_trunc = truncate_text(product_a_details['product_title'])
        product_b_title_trunc = truncate_text(product_b_details['product_title'])

        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            # Sentiment Chart
            counts_a = product_a_reviews['sentiment'].value_counts().reindex(['Positive', 'Neutral', 'Negative']).fillna(0)
            dist_a = counts_a / counts_a.sum() if counts_a.sum() > 0 else counts_a
            counts_b = product_b_reviews['sentiment'].value_counts().reindex(['Positive', 'Neutral', 'Negative']).fillna(0)
            dist_b = counts_b / counts_b.sum() if counts_b.sum() > 0 else counts_b

            df_a = pd.DataFrame({'Proportion': dist_a, 'Count': counts_a}).reset_index(); df_a.columns = ['Sentiment', 'Proportion', 'Count']; df_a['Product'] = product_a_title_trunc
            df_b = pd.DataFrame({'Proportion': dist_b, 'Count': counts_b}).reset_index(); df_b.columns = ['Sentiment', 'Proportion', 'Count']; df_b['Product'] = product_b_title_trunc
            plot_df = pd.concat([df_a, df_b])

            sentiment_chart = alt.Chart(plot_df).mark_bar().encode(
                x=alt.X('Sentiment:N', title="Sentiment", sort=['Positive', 'Neutral', 'Negative']),
                y=alt.Y('Proportion:Q', title="Proportion of Reviews", axis=alt.Axis(format='%')),
                color=alt.Color('Product:N', scale=alt.Scale(range=['#4c78a8', '#f58518'])),
                xOffset='Product:N',
                tooltip=[ alt.Tooltip('Product:N'), alt.Tooltip('Sentiment:N'), alt.Tooltip('Count:Q', title='Review Count'), alt.Tooltip('Proportion:Q', title='Proportion', format='.1%')]
            ).properties(title="Sentiment Comparison")
            st.altair_chart(sentiment_chart, use_container_width=True)

        with chart_col2:
            # Rating Chart
            rating_counts_a = product_a_reviews['rating'].value_counts().reindex([5, 4, 3, 2, 1]).fillna(0)
            rating_dist_a = rating_counts_a / rating_counts_a.sum() if rating_counts_a.sum() > 0 else rating_counts_a
            rating_counts_b = product_b_reviews['rating'].value_counts().reindex([5, 4, 3, 2, 1]).fillna(0)
            rating_dist_b = rating_counts_b / rating_counts_b.sum() if rating_counts_b.sum() > 0 else rating_counts_b

            df_a_ratings = pd.DataFrame({'Proportion': rating_dist_a, 'Count': rating_counts_a}).reset_index(); df_a_ratings.columns = ['Rating', 'Proportion', 'Count']; df_a_ratings['Product'] = product_a_title_trunc
            df_b_ratings = pd.DataFrame({'Proportion': rating_dist_b, 'Count': rating_counts_b}).reset_index(); df_b_ratings.columns = ['Rating', 'Proportion', 'Count']; df_b_ratings['Product'] = product_b_title_trunc
            plot_df_ratings = pd.concat([df_a_ratings, df_b_ratings])

            rating_chart = alt.Chart(plot_df_ratings).mark_bar().encode(
                x=alt.X('Rating:O', title="Star Rating", sort=alt.EncodingSortField(field="Rating", order="descending")),
                y=alt.Y('Proportion:Q', title="Proportion of Reviews", axis=alt.Axis(format='%')),
                color=alt.Color('Product:N', scale=alt.Scale(range=['#4c78a8', '#f58518'])),
                xOffset='Product:N',
                tooltip=[alt.Tooltip('Product:N'), alt.Tooltip('Rating:O'), alt.Tooltip('Count:Q', title='Review Count'), alt.Tooltip('Proportion:Q', title='Proportion', format='.1%')]
            ).properties(title="Rating Comparison")
            st.altair_chart(rating_chart, use_container_width=True)

        # --- Feature-Level Performance: Radar Chart ---
        st.markdown("---")
        st.subheader("Feature-Level Performance Comparison")
        st.info("This radar chart compares the average sentiment score for the most frequently discussed common aspects.")

        aspects_a = extract_aspects_with_sentiment(product_a_reviews)
        aspects_b = extract_aspects_with_sentiment(product_b_reviews)

        if not aspects_a.empty and not aspects_b.empty:
            # (The rest of the radar chart logic remains the same)
            counts_a = aspects_a['aspect'].value_counts()
            counts_b = aspects_b['aspect'].value_counts()
            common_aspects = set(counts_a.index).intersection(set(counts_b.index))

            if len(common_aspects) >= 3:
                total_counts = (counts_a.reindex(common_aspects, fill_value=0) + counts_b.reindex(common_aspects, fill_value=0)).sort_values(ascending=False)

                num_aspects_to_show = st.slider(
                    "Select number of top aspects to display:", min_value=3, max_value=min(20, len(total_counts)),
                    value=min(5, len(total_counts)), key="radar_aspect_slider"
                )

                top_common_aspects = total_counts.nlargest(num_aspects_to_show).index.tolist()

                aspects_a = aspects_a.merge(product_a_reviews[['review_id', 'sentiment_score']], on='review_id')
                aspects_b = aspects_b.merge(product_b_reviews[['review_id', 'sentiment_score']], on='review_id')

                avg_sent_a = aspects_a.groupby('aspect')['sentiment_score'].mean().reindex(top_common_aspects)
                avg_sent_b = aspects_b.groupby('aspect')['sentiment_score'].mean().reindex(top_common_aspects)

                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(r=avg_sent_a.values, theta=avg_sent_a.index, fill='toself', name=product_a_title_trunc, marker_color='#4c78a8', opacity=0.7))
                fig.add_trace(go.Scatterpolar(r=avg_sent_b.values, theta=avg_sent_b.index, fill='toself', name=product_b_title_trunc, marker_color='#f58518', opacity=0.7))
                fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[-1, 1])), showlegend=True, title="Comparative Sentiment Scores by Aspect")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Not enough common aspects (at least 3) found to generate a comparison chart.")
        else:
            st.info("Not enough aspect data for one or both products to generate a comparison.")


if __name__ == "__main__":
    main()
