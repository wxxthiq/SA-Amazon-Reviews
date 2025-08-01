# pages/5_Product_Comparison.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import altair as alt
import json
from utils.database_utils import (
    connect_to_db,
    get_product_details,
    get_reviews_for_product,
    get_filtered_products,
    get_product_date_range,
    get_aspects_for_product
)

# --- Page Config & Constants ---
st.set_page_config(layout="wide", page_title="Product Comparison")
DB_PATH = "amazon_reviews_final.duckdb"
PRODUCTS_PER_PAGE_COMPARE = 9
PLACEHOLDER_IMAGE_URL = "https://via.placeholder.com/200"
conn = connect_to_db(DB_PATH)

# --- Session State Initialization ---
def init_session_state():
    """Initializes all necessary session state variables for this page."""
    if 'product_b_asin' not in st.session_state:
        st.session_state.product_b_asin = None
    if 'compare_search_term' not in st.session_state:
        st.session_state.compare_search_term = ""
    if 'compare_sort_by' not in st.session_state:
        st.session_state.compare_sort_by = "Popularity (Most Reviews)"
    if 'compare_page' not in st.session_state:
        st.session_state.compare_page = 0

def get_sentiment_icon(score):
    """Returns an icon based on the sentiment score."""
    if score is None or pd.isna(score):
        return ""
    if score > 0.3:
        return "😊"
    elif score < -0.3:
        return "😞"
    else:
        return "😐"

def calculate_metrics(product_details, reviews_df):
    """Calculates all performance metrics for a product and returns them in a dictionary."""
    metrics = {
        'avg_rating': product_details.get('average_rating', 0),
        'review_count': len(reviews_df),
        'consensus': "N/A",
        'verified_rate': None,
        'avg_sentiment': None
    }
    if not reviews_df.empty:
        if len(reviews_df) > 1:
            metrics['consensus'] = get_rating_consensus(reviews_df['rating'].std())
        
        metrics['verified_rate'] = (reviews_df['verified_purchase'].sum() / len(reviews_df)) * 100
        
        if 'sentiment_score' in reviews_df.columns:
            metrics['avg_sentiment'] = reviews_df['sentiment_score'].mean()
            
    return metrics
    
def display_product_metadata(column, product_details, metrics, other_metrics, title):
    """Displays metadata and compares metrics against another product."""
    with column:
        st.subheader(title)
        st.markdown(f"**{product_details['product_title']}**")
        st.caption(f"Category: {product_details.get('category', 'N/A')} | Store: {product_details.get('store', 'N/A')}")

        image_urls_str = product_details.get('image_urls')
        image_urls = image_urls_str.split(',') if pd.notna(image_urls_str) and image_urls_str else []
        
        # --- CHANGE 1: Set a fixed height for the image ---
        st.markdown(f'<div class="product-image"><img src="{image_urls[0] if image_urls else PLACEHOLDER_IMAGE_URL}" width="100%"></div>', unsafe_allow_html=True)

        st.markdown("**Performance Summary (based on filters)**")
        
        # --- METRICS WITH DIFFERENTIALS ---
        m_col1, m_col2, m_col3 = st.columns(3)

        # Avg Rating
        delta_rating = None
        if other_metrics and metrics['avg_rating'] is not None and other_metrics['avg_rating'] is not None:
            delta_rating = metrics['avg_rating'] - other_metrics['avg_rating']
        m_col1.metric("Avg. Rating", f"{metrics.get('avg_rating', 0):.2f} ⭐", delta=f"{delta_rating:.2f}" if delta_rating is not None else None)

        # Filtered Reviews
        m_col2.metric("Filtered Reviews", f"{metrics.get('review_count', 0):,} 📝")

        # Reviewer Consensus (no delta for text)
        m_col3.metric("Reviewer Consensus", metrics.get('consensus', 'N/A'))
        
        m_col4, m_col5 = st.columns(2)
        
        # Verified Rate
        delta_verified = None
        if other_metrics and metrics['verified_rate'] is not None and other_metrics['verified_rate'] is not None:
            delta_verified = metrics['verified_rate'] - other_metrics['verified_rate']
        m_col4.metric("Verified Rate", f"{metrics.get('verified_rate', 0):.1f}%", delta=f"{delta_verified:.1f}%" if delta_verified is not None else None)

        # Avg Sentiment
        sentiment_icon = get_sentiment_icon(metrics.get('avg_sentiment'))
        delta_sentiment = None
        if other_metrics and metrics['avg_sentiment'] is not None and other_metrics['avg_sentiment'] is not None:
            delta_sentiment = metrics['avg_sentiment'] - other_metrics['avg_sentiment']
        
        # --- CHANGE 2: Moved the icon from the label to the value string ---
        m_col5.metric("Avg. Sentiment", f"{metrics.get('avg_sentiment', 0):.2f} {sentiment_icon}", delta=f"{delta_sentiment:.2f}" if delta_sentiment is not None else None)


        # --- CONSOLIDATED PRODUCT SPECIFICATIONS ---
        with st.expander("View Product Specifications"):
            # (This part remains the same)
            if pd.notna(product_details.get('description')):
                st.markdown("---")
                st.markdown("**Description**")
                st.write(product_details['description'])
            if pd.notna(product_details.get('features')):
                st.markdown("---")
                st.markdown("**Features**")
                try:
                    features_list = json.loads(product_details['features']) if isinstance(product_details['features'], str) else product_details['features']
                    if features_list:
                        for feature in features_list:
                            st.markdown(f"- {feature}")
                except (json.JSONDecodeError, TypeError):
                    st.write("Could not parse features.")
            if pd.notna(product_details.get('details')):
                st.markdown("---")
                st.markdown("**Technical Details**")
                try:
                    details_dict = json.loads(product_details['details']) if isinstance(product_details['details'], str) else product_details['details']
                    if details_dict:
                        st.json(details_dict)
                except (json.JSONDecodeError, TypeError):
                    st.write("Could not parse product details.")

        if st.session_state.product_b_asin and title == "Comparison Product":
            if st.button("Change Comparison Product", use_container_width=True, key="change_product_b"):
                st.session_state.product_b_asin = None
                st.session_state.compare_page = 0
                st.rerun()
                
def show_product_selection_pane(column, category, product_a_asin):
    """Displays the UI for searching, sorting, and selecting a product."""
    with column:
        st.subheader("Select a Product to Compare")
        st.info(f"Browse products in the same category: **{category}**")

        # --- Search and Sort Controls ---
        c1, c2 = st.columns([2,1])
        with c1:
            st.session_state.compare_search_term = st.text_input(
                "Search by product title:",
                value=st.session_state.compare_search_term,
                key="compare_search"
            )
        with c2:
            st.session_state.compare_sort_by = st.selectbox(
                "Sort By",
                ["Popularity (Most Reviews)", "Highest Rating", "Lowest Rating"],
                key="compare_sort"
            )

        # --- Fetch and Display Products ---
        products_df, total_count = get_filtered_products(
            _conn=conn,
            category=category,
            search_term=st.session_state.compare_search_term,
            sort_by=st.session_state.compare_sort_by,
            rating_range=None,
            review_count_range=None,
            limit=PRODUCTS_PER_PAGE_COMPARE,
            offset=st.session_state.compare_page * PRODUCTS_PER_PAGE_COMPARE
        )
        products_df = products_df[products_df['parent_asin'] != product_a_asin]

        if products_df.empty:
            st.warning("No other products found matching your criteria.")
            return

        # --- Product Grid ---
        for i in range(0, len(products_df), 3):
            cols = st.columns(3)
            for j, col in enumerate(cols):
                if i + j < len(products_df):
                    product = products_df.iloc[i+j]
                    with col.container(border=True):
                        img_url = product.get('image_urls', '').split(',')[0] if pd.notna(product.get('image_urls')) else PLACEHOLDER_IMAGE_URL
                        st.image(img_url, use_container_width=True)
                        st.markdown(f"<small>{product['product_title']}</small>", unsafe_allow_html=True)
                        
                        # --- NEW: Display rating and review count ---
                        avg_rating = product.get('average_rating', 0)
                        review_count = product.get('review_count', 0)
                        st.caption(f"{avg_rating:.2f} ⭐ | {int(review_count)} reviews")

                        if st.button("Select to Compare", key=f"select_{product['parent_asin']}", use_container_width=True):
                            st.session_state.product_b_asin = product['parent_asin']
                            st.rerun()

        # --- Pagination ---
        st.markdown("---")
        total_pages = (total_count + PRODUCTS_PER_PAGE_COMPARE - 1) // PRODUCTS_PER_PAGE_COMPARE
        if total_pages > 1:
            p_col1, p_col2, p_col3 = st.columns([1,1,1])
            with p_col1:
                if st.session_state.compare_page > 0:
                    if st.button("⬅️ Previous"):
                        st.session_state.compare_page -= 1
                        st.rerun()
            with p_col2:
                st.write(f"Page {st.session_state.compare_page + 1} of {total_pages}")
            with p_col3:
                if (st.session_state.compare_page + 1) < total_pages:
                    if st.button("Next ➡️"):
                        st.session_state.compare_page += 1
                        st.rerun()

# --- Data & Charting Helper Functions ---
def get_rating_consensus(std_dev):
    """Interprets standard deviation of ratings into a consensus label."""
    if std_dev is None or pd.isna(std_dev): return "N/A"
    if std_dev < 1.1: return "✅ Consistent"
    elif std_dev < 1.4: return "↔️ Mixed"
    else: return "⚠️ Polarizing"

def truncate_text(text, max_length=15):
    """Truncates text for chart labels."""
    return text if len(text) <= max_length else text[:max_length] + "..."

# --- Main App Logic ---
def main():
    st.title("⚖️ Product Comparison")

    st.markdown("""
        <style>
        .product-image-container {
            width: 100%;
            height: 250px; /* Explicitly set the container height */
            display: flex;
            justify-content: center;
            align-items: center;
            margin-bottom: 1rem; /* Add some space below the image container */
        }
        .product-image-container img {
            max-width: 100%;
            max-height: 100%;
            object-fit: contain;
        }
        </style>
    """, unsafe_allow_html=True)
    init_session_state()

    if st.button("⬅️ Back to Sentiment Overview"):
        st.switch_page("pages/1_Sentiment_Overview.py")

    if 'selected_product' not in st.session_state or st.session_state.selected_product is None:
        st.warning("Please select a product from the main search page first to begin a comparison.")
        st.stop()

    product_a_asin = st.session_state.selected_product
    product_a_details = get_product_details(conn, product_a_asin).iloc[0]

    # --- Sidebar Filters for Comparison ---
    st.sidebar.header("🔬 Comparison Filters")
    # Set a default date range to avoid errors if one product has no reviews
    min_date_a, max_date_a = get_product_date_range(conn, product_a_asin)
    min_date_b, max_date_b = (min_date_a, max_date_a) # Default to A's dates
    if st.session_state.product_b_asin:
        min_date_b, max_date_b = get_product_date_range(conn, st.session_state.product_b_asin)
    
    selected_date_range = st.sidebar.date_input("Filter by Date Range", value=(min(min_date_a, min_date_b), max(max_date_a, max_date_b)), key='compare_date_filter')
    selected_ratings = st.sidebar.multiselect("Filter by Star Rating", options=[1, 2, 3, 4, 5], default=[1, 2, 3, 4, 5], key='compare_rating_filter')
    selected_sentiments = st.sidebar.multiselect("Filter by Sentiment", options=['Positive', 'Negative', 'Neutral'], default=['Positive', 'Negative', 'Neutral'], key='compare_sentiment_filter')
    selected_verified = st.sidebar.radio("Filter by Purchase Status", ["All", "Verified Only", "Not Verified"], index=0, key='compare_verified_filter')

    # --- Load Data for Product A ---
    product_a_reviews = get_reviews_for_product(conn, product_a_asin, selected_date_range, tuple(selected_ratings), tuple(selected_sentiments), selected_verified)

    # --- Main Two-Column Layout ---
    col1, col2 = st.columns(2)

    # --- Calculate metrics for Product A ---
    metrics_a = calculate_metrics(product_a_details, product_a_reviews)

    if not st.session_state.product_b_asin:
        # If no product B, display A's metadata without comparison.
        # The 'other_metrics' argument is passed as None.
        display_product_metadata(col1, product_a_details, metrics_a, None, "Original Product")
        show_product_selection_pane(col2, product_a_details['category'], product_a_asin)
    else:
        # --- Product B is Selected, Calculate its metrics and then display both ---
        product_b_asin = st.session_state.product_b_asin
        product_b_details = get_product_details(conn, product_b_asin).iloc[0]
        product_b_reviews = get_reviews_for_product(conn, product_b_asin, selected_date_range, tuple(selected_ratings), tuple(selected_sentiments), selected_verified)
        
        metrics_b = calculate_metrics(product_b_details, product_b_reviews)

        # Display both products, passing the other's metrics for comparison.
        display_product_metadata(col1, product_a_details, metrics_a, metrics_b, "Original Product")
        display_product_metadata(col2, product_b_details, metrics_b, metrics_a, "Comparison Product")

        
        # --- RENDER COMPARISON CHARTS BELOW THE METADATA ---
        st.markdown("---")
        st.subheader("📊 At-a-Glance Comparison")
        st.info("These charts directly compare the proportion of sentiments and star ratings for each product based on your filters. Hover over the bars to see the raw counts.")

        chart_col1, chart_col2 = st.columns(2)
        
        # --- Sentiment Comparison Chart ---
        with chart_col1:
            counts_a = product_a_reviews['sentiment'].value_counts().reindex(['Positive', 'Neutral', 'Negative']).fillna(0)
            dist_a = counts_a / counts_a.sum() if counts_a.sum() > 0 else counts_a
            counts_b = product_b_reviews['sentiment'].value_counts().reindex(['Positive', 'Neutral', 'Negative']).fillna(0)
            dist_b = counts_b / counts_b.sum() if counts_b.sum() > 0 else counts_b
            
            df_a = pd.DataFrame({'Proportion': dist_a, 'Count': counts_a}).reset_index(); df_a.columns = ['Sentiment', 'Proportion', 'Count']; df_a['Product'] = truncate_text(product_a_details['product_title'])
            df_b = pd.DataFrame({'Proportion': dist_b, 'Count': counts_b}).reset_index(); df_b.columns = ['Sentiment', 'Proportion', 'Count']; df_b['Product'] = truncate_text(product_b_details['product_title'])
            plot_df = pd.concat([df_a, df_b])

            sentiment_chart = alt.Chart(plot_df).mark_bar().encode(
                x=alt.X('Sentiment:N', title="Sentiment", sort=['Positive', 'Neutral', 'Negative']),
                y=alt.Y('Proportion:Q', title="Proportion of Reviews", axis=alt.Axis(format='%')),
                color=alt.Color('Product:N', scale=alt.Scale(range=['#4c78a8', '#f58518'])),
                xOffset='Product:N',
                tooltip=[alt.Tooltip('Product:N'), alt.Tooltip('Sentiment:N'), alt.Tooltip('Count:Q', title='Review Count'), alt.Tooltip('Proportion:Q', title='Proportion', format='.1%')]
            ).properties(title="Sentiment Comparison")
            st.altair_chart(sentiment_chart, use_container_width=True)

        # --- Rating Comparison Chart ---
        with chart_col2:
            rating_counts_a = product_a_reviews['rating'].value_counts().reindex([5, 4, 3, 2, 1]).fillna(0)
            rating_dist_a = rating_counts_a / rating_counts_a.sum() if rating_counts_a.sum() > 0 else rating_counts_a
            rating_counts_b = product_b_reviews['rating'].value_counts().reindex([5, 4, 3, 2, 1]).fillna(0)
            rating_dist_b = rating_counts_b / rating_counts_b.sum() if rating_counts_b.sum() > 0 else rating_counts_b
            
            df_a_ratings = pd.DataFrame({'Proportion': rating_dist_a, 'Count': rating_counts_a}).reset_index(); df_a_ratings.columns = ['Rating', 'Proportion', 'Count']; df_a_ratings['Product'] = truncate_text(product_a_details['product_title'])
            df_b_ratings = pd.DataFrame({'Proportion': rating_dist_b, 'Count': rating_counts_b}).reset_index(); df_b_ratings.columns = ['Rating', 'Proportion', 'Count']; df_b_ratings['Product'] = truncate_text(product_b_details['product_title'])
            plot_df_ratings = pd.concat([df_a_ratings, df_b_ratings])

            rating_chart = alt.Chart(plot_df_ratings).mark_bar().encode(
                x=alt.X('Rating:O', title="Star Rating", sort=alt.EncodingSortField(field="Rating", order="descending")),
                y=alt.Y('Proportion:Q', title="Proportion of Reviews", axis=alt.Axis(format='%')),
                color=alt.Color('Product:N', scale=alt.Scale(range=['#4c78a8', '#f58518'])),
                xOffset='Product:N',
                tooltip=[alt.Tooltip('Product:N'), alt.Tooltip('Rating:O'), alt.Tooltip('Count:Q', title='Review Count'), alt.Tooltip('Proportion:Q', title='Proportion', format='.1%')]
            ).properties(title="Rating Comparison")
            st.altair_chart(rating_chart, use_container_width=True)
            
        # --- Feature-Level Performance: Comparative Radar Chart ---
        st.markdown("---")
        st.subheader("🔎 Feature-Level Performance Comparison")
        st.info("This radar chart compares the average sentiment score for the most frequently discussed common aspects. A score closer to 1 is more positive, and a score closer to -1 is more negative.")

        aspects_a = get_aspects_for_product(conn, product_a_asin, selected_date_range, tuple(selected_ratings), tuple(selected_sentiments), selected_verified)
        aspects_b = get_aspects_for_product(conn, product_b_asin, selected_date_range, tuple(selected_ratings), tuple(selected_sentiments), selected_verified)
        
        if not aspects_a.empty and not aspects_b.empty:
            # Merge with reviews to get sentiment scores
            aspects_a = aspects_a.merge(product_a_reviews[['review_id', 'sentiment_score']], on='review_id', how='inner')
            aspects_b = aspects_b.merge(product_b_reviews[['review_id', 'sentiment_score']], on='review_id', how='inner')

            counts_a = aspects_a['aspect'].value_counts()
            counts_b = aspects_b['aspect'].value_counts()
            common_aspects = set(counts_a.index).intersection(set(counts_b.index))
            
            if len(common_aspects) >= 3:
                total_counts = (counts_a.reindex(common_aspects, fill_value=0) + counts_b.reindex(common_aspects, fill_value=0)).sort_values(ascending=False)
                
                num_aspects_to_show = st.slider(
                    "Select number of top aspects to display:",
                    min_value=3, max_value=min(20, len(total_counts)), value=min(5, len(total_counts)),
                    key="radar_aspect_slider"
                )
                top_common_aspects = total_counts.nlargest(num_aspects_to_show).index.tolist()

                avg_sent_a = aspects_a[aspects_a['aspect'].isin(top_common_aspects)].groupby('aspect')['sentiment_score'].mean().reindex(top_common_aspects)
                avg_sent_b = aspects_b[aspects_b['aspect'].isin(top_common_aspects)].groupby('aspect')['sentiment_score'].mean().reindex(top_common_aspects)

                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(r=avg_sent_a.values, theta=avg_sent_a.index, fill='toself', name=truncate_text(product_a_details['product_title']), marker_color='#4c78a8', opacity=0.7))
                fig.add_trace(go.Scatterpolar(r=avg_sent_b.values, theta=avg_sent_b.index, fill='toself', name=truncate_text(product_b_details['product_title']), marker_color='#f58518', opacity=0.7))
                fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[-1, 1])), showlegend=True)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Not enough common aspects (at least 3) found between the products with the current filters to generate a comparison chart.")
        else:
            st.warning("Not enough aspect data for one or both products to generate a comparison with the current filters.")

if __name__ == "__main__":
    main()
