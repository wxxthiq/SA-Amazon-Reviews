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
        
       # --- METRICS WITH HELP ICONS ---
        m_col1, m_col2, m_col3 = st.columns(3)

        # Avg Rating
        delta_rating = None
        if other_metrics and metrics['avg_rating'] is not None and other_metrics['avg_rating'] is not None:
            delta_rating = metrics['avg_rating'] - other_metrics['avg_rating']
        m_col1.metric("Avg. Rating", f"{metrics.get('avg_rating', 0):.2f} ⭐", delta=f"{delta_rating:.2f}" if delta_rating is not None else None,
                      help="The average star rating from all reviews for this product.")

        # Filtered Reviews
        m_col2.metric("Filtered Reviews", f"{metrics.get('review_count', 0):,} 📝",
                      help="The number of reviews that match the current filter settings in the sidebar.")

        # Reviewer Consensus
        m_col3.metric("Reviewer Consensus", metrics.get('consensus', 'N/A'),
                      help="Measures agreement in star ratings. 'Consistent' means ratings are similar, while 'Polarizing' suggests many high and low ratings with few in the middle.")
        
        m_col4, m_col5 = st.columns(2)
        
        # Verified Rate
        delta_verified = None
        if other_metrics and metrics['verified_rate'] is not None and other_metrics['verified_rate'] is not None:
            delta_verified = metrics['verified_rate'] - other_metrics['verified_rate']
        m_col4.metric("Verified Rate", f"{metrics.get('verified_rate', 0):.1f}%", delta=f"{delta_verified:.1f}%" if delta_verified is not None else None,
                      help="The percentage of filtered reviews that are from 'Verified Purchases'.")

        # Avg Sentiment
        sentiment_icon = get_sentiment_icon(metrics.get('avg_sentiment'))
        delta_sentiment = None
        if other_metrics and metrics['avg_sentiment'] is not None and other_metrics['avg_sentiment'] is not None:
            delta_sentiment = metrics['avg_sentiment'] - other_metrics['avg_sentiment']
        m_col5.metric("Avg. Sentiment", f"{metrics.get('avg_sentiment', 0):.2f} {sentiment_icon}", delta=f"{delta_sentiment:.2f}" if delta_sentiment is not None else None,
                      help="The average sentiment score of review text, from -1 (very negative) to +1 (very positive).")


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
                # --- FEATURE-LEVEL PERFORMANCE (FINAL REVISION) ---
        st.markdown("---")
        st.subheader("🔎 Feature-Level Performance")
        st.info(
            "This section compares sentiment towards features (aspects) that are **common to both products**. "
            "Use the filters to adjust the comparison and the charts to analyze the results."
        )

        aspects_a = get_aspects_for_product(conn, product_a_asin, selected_date_range, tuple(selected_ratings), tuple(selected_sentiments), selected_verified)
        aspects_b = get_aspects_for_product(conn, product_b_asin, selected_date_range, tuple(selected_ratings), tuple(selected_sentiments), selected_verified)

        if not aspects_a.empty and not aspects_b.empty:
            # First, find the common aspects
            aspects_a_set = set(aspects_a['aspect'])
            aspects_b_set = set(aspects_b['aspect'])
            common_aspects_list = list(aspects_a_set.intersection(aspects_b_set))

            if not common_aspects_list:
                st.warning("No common features were found between these two products with the current filters.")
                st.stop()

            # Filter the original dataframes to only include common aspects
            common_aspects_a = aspects_a[aspects_a['aspect'].isin(common_aspects_list)].copy()
            common_aspects_b = aspects_b[aspects_b['aspect'].isin(common_aspects_list)].copy()

            # Add product titles for faceting and combine
            common_aspects_a['Product'] = product_a_title
            common_aspects_b['Product'] = product_b_title
            combined_common_aspects_df = pd.concat([common_aspects_a, common_aspects_b])

            # --- Create a two-column layout for controls and charts ---
            control_col, chart_col = st.columns([1, 2])

            with control_col:
                st.markdown("**Comparison Controls**")
                sort_option = st.selectbox(
                    "Sort aspects by:",
                    ("Total Mentions", "Most Positive", "Most Negative"),
                    key="aspect_sort_selector"
                )
                
                # --- ADDED MENTION THRESHOLD SLIDER ---
                total_counts = combined_common_aspects_df['aspect'].value_counts()
                max_threshold = int(total_counts.max()) if not total_counts.empty else 1
                min_mentions = st.slider(
                    "Minimum Mention Threshold:", 1, max(1, max_threshold), 1,
                    key="aspect_mention_threshold",
                    help="Only show common aspects that have at least this many total mentions (across both products)."
                )

            # --- Data processing based on controls ---
            aspects_to_keep = total_counts[total_counts >= min_mentions].index
            
            # Create pivot for sorting
            pivot_df = combined_common_aspects_df.groupby(['aspect', 'sentiment']).size().unstack(fill_value=0)
            if 'Positive' not in pivot_df: pivot_df['Positive'] = 0
            if 'Negative' not in pivot_df: pivot_df['Negative'] = 0
            pivot_df['total'] = pivot_df.sum(axis=1)
            pivot_df['positive_pct'] = pivot_df['Positive'] / pivot_df['total'] if 'Positive' in pivot_df else 0

            # Filter pivot by mention threshold and sort
            pivot_df = pivot_df[pivot_df.index.isin(aspects_to_keep)]
            if sort_option == "Most Positive":
                sorted_aspects = pivot_df.sort_values('positive_pct', ascending=False).index
            elif sort_option == "Most Negative":
                sorted_aspects = pivot_df.sort_values('Negative', ascending=False).index
            else: # Total Mentions
                sorted_aspects = pivot_df.sort_values('total', ascending=False).index
            
            # Final dataframe for charting
            chart_df = combined_common_aspects_df[combined_common_aspects_df['aspect'].isin(sorted_aspects)]


            with chart_col:
                st.markdown("**Common Aspect Summary**")
                if not chart_df.empty:
                    # --- Grouped Stacked Bar Chart on COMMON aspects ---
                    aspect_summary_chart = alt.Chart(chart_df).mark_bar().encode(
                        x=alt.X('count()', stack='normalize', axis=alt.Axis(title='Sentiment Distribution', format='%')),
                        y=alt.Y('aspect:N', title=None, sort=alt.EncodingSortField(field="aspect", op="count", order='descending')),
                        color=alt.Color('sentiment:N', scale=alt.Scale(domain=['Positive', 'Neutral', 'Negative'], range=['#1a9850', '#cccccc', '#d73027']), legend=None),
                        row=alt.Row('Product:N', title=None, header=alt.Header(labelOrient='top', labelPadding=10)),
                        tooltip=['Product', 'aspect', 'sentiment', alt.Tooltip('count()', title='Mentions')]
                    ).properties(height=150)

                    st.altair_chart(aspect_summary_chart, use_container_width=True)
                else:
                    st.warning("No aspects meet the current mention threshold. Please lower the threshold.")

            # --- Radar chart for direct comparison ---
            st.markdown("**Direct Sentiment Score Comparison**")
            aspects_a_with_scores = common_aspects_a.merge(product_a_reviews[['review_id', 'sentiment_score']], on='review_id', how='inner')
            aspects_b_with_scores = common_aspects_b.merge(product_b_reviews[['review_id', 'sentiment_score']], on='review_id', how='inner')

            selected_for_radar = st.multiselect(
                "Select from common, filtered features for radar chart:",
                options=sorted_aspects.tolist(),
                default=sorted_aspects.tolist()[:5] # Default to top 5 sorted aspects
            )

            if selected_for_radar:
                avg_sent_a = aspects_a_with_scores[aspects_a_with_scores['aspect'].isin(selected_for_radar)].groupby('aspect')['sentiment_score'].mean().reindex(selected_for_radar)
                avg_sent_b = aspects_b_with_scores[aspects_b_with_scores['aspect'].isin(selected_for_radar)].groupby('aspect')['sentiment_score'].mean().reindex(selected_for_radar)
                
                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(r=avg_sent_a.values, theta=avg_sent_a.index, fill='toself', name=product_a_title, marker_color='#4c78a8', opacity=0.7))
                fig.add_trace(go.Scatterpolar(r=avg_sent_b.values, theta=avg_sent_b.index, fill='toself', name=product_b_title, marker_color='#f58518', opacity=0.7))
                fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[-1, 1])), showlegend=True, margin=dict(l=40, r=40, t=40, b=40))
                st.plotly_chart(fig, use_container_width=True)

        else:
            st.warning("Not enough aspect data for one or both products to generate a feature-level comparison.")
            
if __name__ == "__main__":
    main()
