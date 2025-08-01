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
        # --- RENDER COMPARISON CHARTS BELOW THE METADATA ---
        st.markdown("---")
        st.subheader("📊 At-a-Glance Comparison")
        st.info(
            "These charts show the proportional breakdown of ratings and sentiments for each product, based on your filters. "
            "This allows for a fair comparison even if the number of reviews is different."
        )

        # --- Prepare data for charts ---
        # Add a 'Product' column to each dataframe before combining
        product_a_title = truncate_text(product_a_details['product_title'])
        product_b_title = truncate_text(product_b_details['product_title'])
        product_a_reviews['Product'] = product_a_title
        product_b_reviews['Product'] = product_b_title

        combined_reviews_df = pd.concat([product_a_reviews, product_b_reviews], ignore_index=True)

        # --- Create a two-column layout for the charts ---
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            st.markdown("**⭐ Rating Distribution**")
            if not combined_reviews_df.empty:
                # Create a categorical rating string for sorting and labeling
                combined_reviews_df['rating_cat'] = combined_reviews_df['rating'].astype(str) + ' ⭐'
                
                rating_chart = alt.Chart(combined_reviews_df).mark_bar().encode(
                    x=alt.X('count()', stack='normalize', axis=alt.Axis(title='Percentage', format='%')),
                    color=alt.Color('rating_cat:N',
                                    scale=alt.Scale(domain=['5 ⭐', '4 ⭐', '3 ⭐', '2 ⭐', '1 ⭐'],
                                                    range=['#2ca02c', '#98df8a', '#ffdd71', '#ff9896', '#d62728']),
                                    legend=alt.Legend(title="Rating")),
                    order=alt.Order('rating_cat', sort='descending'),
                    tooltip=[
                        alt.Tooltip('Product:N', title='Product'),
                        alt.Tooltip('rating_cat:N', title='Rating'),
                        alt.Tooltip('count()', title='Review Count')
                    ]
                ).properties(
                    height=80
                ).facet(
                    row=alt.Row('Product:N', title=None, header=alt.Header(labelOrient='top', labelPadding=10))
                )
                st.altair_chart(rating_chart, use_container_width=True)

        with chart_col2:
            st.markdown("**😊 Sentiment Distribution**")
            if not combined_reviews_df.empty:
                sentiment_chart = alt.Chart(combined_reviews_df).mark_bar().encode(
                    x=alt.X('count()', stack='normalize', axis=alt.Axis(title='Percentage', format='%')),
                    color=alt.Color('sentiment:N',
                                    scale=alt.Scale(domain=['Positive', 'Neutral', 'Negative'],
                                                    range=['#1a9850', '#cccccc', '#d73027']),
                                    legend=alt.Legend(title="Sentiment")),
                    order=alt.Order('sentiment', sort='descending'),
                     tooltip=[
                        alt.Tooltip('Product:N', title='Product'),
                        alt.Tooltip('sentiment:N', title='Sentiment'),
                        alt.Tooltip('count()', title='Review Count')
                    ]
                ).properties(
                    height=80
                ).facet(
                    row=alt.Row('Product:N', title=None, header=alt.Header(labelOrient='top', labelPadding=10))
                )
                st.altair_chart(sentiment_chart, use_container_width=True)
        
        # --- FEATURE-LEVEL PERFORMANCE (IMPROVED LOGIC) ---
        st.markdown("---")
        st.subheader("🔎 Feature-Level Performance")
        
        aspects_a = get_aspects_for_product(conn, product_a_asin, selected_date_range, tuple(selected_ratings), tuple(selected_sentiments), selected_verified)
        aspects_b = get_aspects_for_product(conn, product_b_asin, selected_date_range, tuple(selected_ratings), tuple(selected_sentiments), selected_verified)

        if not aspects_a.empty and not aspects_b.empty:
            aspects_a = aspects_a.merge(product_a_reviews[['review_id', 'sentiment_score', 'text', 'helpful_vote']], on='review_id', how='inner')
            aspects_b = aspects_b.merge(product_b_reviews[['review_id', 'sentiment_score', 'text', 'helpful_vote']], on='review_id', how='inner')

            # --- 1. Interactive Comparison on Common Features ---
            counts_a = aspects_a['aspect'].value_counts()
            counts_b = aspects_b['aspect'].value_counts()
            common_aspects_list = sorted(list(set(counts_a.index).intersection(set(counts_b.index))))

            st.markdown("**Direct Comparison on Common Features**")
            st.info("Use the dropdown below to add or remove features from the radar chart comparison.")

            if len(common_aspects_list) >= 3:
                # --- REPLACED SLIDER WITH MULTISELECT ---
                top_5_common = (counts_a + counts_b).reindex(common_aspects_list).nlargest(5).index.tolist()
                selected_aspects = st.multiselect(
                    "Select features to compare:",
                    options=common_aspects_list,
                    default=top_5_common,
                    key="aspect_multiselect"
                )

                if len(selected_aspects) >= 1:
                    avg_sent_a = aspects_a[aspects_a['aspect'].isin(selected_aspects)].groupby('aspect')['sentiment_score'].mean().reindex(selected_aspects)
                    avg_sent_b = aspects_b[aspects_b['aspect'].isin(selected_aspects)].groupby('aspect')['sentiment_score'].mean().reindex(selected_aspects)

                    fig = go.Figure()
                    fig.add_trace(go.Scatterpolar(r=avg_sent_a.values, theta=avg_sent_a.index, fill='toself', name=product_a_title, marker_color='#4c78a8', opacity=0.7))
                    fig.add_trace(go.Scatterpolar(r=avg_sent_b.values, theta=avg_sent_b.index, fill='toself', name=product_b_title, marker_color='#f58518', opacity=0.7))
                    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[-1, 1])), showlegend=True, height=400)
                    st.plotly_chart(fig, use_container_width=True)

                    # --- NEW FEATURE: FEATURE DEEP DIVE ---
                    st.markdown("**Feature Deep Dive**")
                    st.info("Select a single feature from your comparison to see relevant review snippets.")
                    
                    deep_dive_aspect = st.selectbox(
                        "Choose a feature to see example reviews:",
                        options=selected_aspects,
                        index=0,
                        key="deep_dive_selector"
                    )

                    if deep_dive_aspect:
                        dd_col1, dd_col2 = st.columns(2)
                        
                        # Function to get and display review snippets
                        def display_snippets(column, product_title, aspect_df, aspect_name):
                            with column:
                                st.markdown(f"**{product_title}**")
                                relevant_reviews = aspect_df[aspect_df['aspect'] == aspect_name].sort_values('helpful_vote', ascending=False)
                                
                                pos_reviews = relevant_reviews[relevant_reviews['sentiment_score'] > 0.3]
                                neg_reviews = relevant_reviews[relevant_reviews['sentiment_score'] < -0.3]

                                st.success("👍 Most Helpful Positive Snippet:")
                                if not pos_reviews.empty:
                                    st.markdown(f"> *{pos_reviews.iloc[0]['text'][:200]}...*")
                                else:
                                    st.caption("No positive reviews found for this aspect.")

                                st.error("👎 Most Helpful Negative Snippet:")
                                if not neg_reviews.empty:
                                    st.markdown(f"> *{neg_reviews.iloc[0]['text'][:200]}...*")
                                else:
                                    st.caption("No negative reviews found for this aspect.")

                        display_snippets(dd_col1, product_a_title, aspects_a, deep_dive_aspect)
                        display_snippets(dd_col2, product_b_title, aspects_b, deep_dive_aspect)

            else:
                st.warning("Not enough common aspects (at least 3) found between the products to generate a comparison chart.")

            # --- 2. Find Unique Aspects ---
            st.markdown("**Unique Differentiators**")
            st.info("These are the most frequently mentioned positive and negative features for one product that are **not** mentioned for the other.")

            unique_a = set(counts_a.index) - set(counts_b.index)
            unique_b = set(counts_b.index) - set(counts_a.index)
            
            u_col1, u_col2 = st.columns(2)
            with u_col1:
                st.markdown(f"**For: {product_a_title}**")
                if unique_a:
                    unique_a_sentiments = aspects_a[aspects_a['aspect'].isin(unique_a)].groupby('aspect')['sentiment_score'].mean()
                    st.success("**Top Unique Strengths**")
                    st.dataframe(unique_a_sentiments.nlargest(3).reset_index(), use_container_width=True, hide_index=True)
                    st.error("**Top Unique Weaknesses**")
                    st.dataframe(unique_a_sentiments.nsmallest(3).reset_index(), use_container_width=True, hide_index=True)
                else:
                    st.write("No unique aspects found.")
            with u_col2:
                st.markdown(f"**For: {product_b_title}**")
                if unique_b:
                    unique_b_sentiments = aspects_b[aspects_b['aspect'].isin(unique_b)].groupby('aspect')['sentiment_score'].mean()
                    st.success("**Top Unique Strengths**")
                    st.dataframe(unique_b_sentiments.nlargest(3).reset_index(), use_container_width=True, hide_index=True)
                    st.error("**Top Unique Weaknesses**")
                    st.dataframe(unique_b_sentiments.nsmallest(3).reset_index(), use_container_width=True, hide_index=True)
                else:
                    st.write("No unique aspects found.")

        else:
            st.warning("Not enough aspect data for one or both products to generate a feature-level comparison.")

if __name__ == "__main__":
    main()
