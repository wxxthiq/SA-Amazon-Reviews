# app.py (Main Search Page)
import streamlit as st
import pandas as pd
from utils.database_utils import connect_to_db, get_all_categories, get_filtered_products, a_download_data_with_versioning

# --- App Configuration ---
DB_PATH = "amazon_reviews_top100.duckdb"
DB_VERSION = 1
PRODUCTS_PER_PAGE = 16
PLACEHOLDER_IMAGE_URL = "https://via.placeholder.com/200"
KAGGLE_DATASET_SLUG = "wathiqsoualhi/amazon-reviews-duckdb-top100"

# --- Page Setup ---
st.set_page_config(layout="wide", page_title="Amazon Review Search")
st.title("🔎 Amazon Product Search")
st.info("This app showcases analysis on the Top 100 most-reviewed products across selected Amazon categories.")

# --- Data Loading ---
a_download_data_with_versioning(KAGGLE_DATASET_SLUG, DB_PATH, DB_VERSION)
conn = connect_to_db(DB_PATH)

# --- Session State Initialization ---
if 'page' not in st.session_state: st.session_state.page = 0
if 'selected_product' not in st.session_state: st.session_state.selected_product = None
if 'category' not in st.session_state: st.session_state.category = "--- Select a Category ---"
if 'search_term' not in st.session_state: st.session_state.search_term = ""
if 'sort_by' not in st.session_state: st.session_state.sort_by = "Popularity (Most Reviews)"
# --- NEW: Initialize session state for new filters ---
if 'rating_range' not in st.session_state: st.session_state.rating_range = (1.0, 5.0)
if 'review_count_range' not in st.session_state: st.session_state.review_count_range = (0, 50000)


# --- Search and Filter UI ---
col1, col2, col3 = st.columns(3)
with col1:
    st.session_state.search_term = st.text_input("Search by product title:", value=st.session_state.search_term)
with col2:
    available_categories = get_all_categories(conn)
    def on_category_change():
        st.session_state.page = 0
    st.session_state.category = st.selectbox("Filter by Category", available_categories, index=available_categories.index(st.session_state.category), on_change=on_category_change)
with col3:
    st.session_state.sort_by = st.selectbox("Sort By", ["Popularity (Most Reviews)", "Highest Rating", "Lowest Rating"], index=["Popularity (Most Reviews)", "Highest Rating", "Lowest Rating"].index(st.session_state.sort_by))

# --- NEW: Advanced Filters UI ---
with st.expander("✨ Advanced Filters"):
    adv_col1, adv_col2 = st.columns(2)
    with adv_col1:
        st.session_state.rating_range = st.slider(
            "Filter by Average Rating:",
            min_value=1.0,
            max_value=5.0,
            value=st.session_state.rating_range,
            step=0.1
        )
    with adv_col2:
        st.session_state.review_count_range = st.slider(
            "Filter by Number of Reviews:",
            min_value=0,
            max_value=50000, # You can adjust this max value based on your dataset
            value=st.session_state.review_count_range,
            step=100
        )


# --- Product Display Logic ---
if st.session_state.category == "--- Select a Category ---":
    st.warning("Please select a category to view products.")
else:
    # --- UPDATED: Pass new filter values to the function ---
    paginated_results, total_results = get_filtered_products(
        _conn=conn,
        category=st.session_state.category,
        search_term=st.session_state.search_term,
        sort_by=st.session_state.sort_by,
        rating_range=st.session_state.rating_range,
        review_count_range=st.session_state.review_count_range,
        limit=PRODUCTS_PER_PAGE,
        offset=st.session_state.page * PRODUCTS_PER_PAGE
    )

    st.markdown("---")
    st.header(f"Found {total_results} Products in '{st.session_state.category}'")

    if paginated_results.empty:
        st.warning("No products match your search criteria on this page.")
    else:
        # Display products in a grid
        for i in range(0, len(paginated_results), 4):
            cols = st.columns(4)
            for j, col in enumerate(cols):
                if i + j < len(paginated_results):
                    row = paginated_results.iloc[i+j]
                    with col.container(border=True):
                        image_urls_str = row.get('image_urls')
                        thumbnail_url = image_urls_str.split(',')[0] if pd.notna(image_urls_str) else PLACEHOLDER_IMAGE_URL
                        st.image(thumbnail_url, use_container_width=True)
                        st.markdown(f"**{row['product_title']}**")
                        avg_rating = row.get('average_rating', 0)
                        review_count = row.get('review_count', 0)
                        st.caption(f"Avg. Rating: {avg_rating:.2f} ⭐ ({int(review_count)} reviews)")
                        
                        if st.button("View Details", key=row['parent_asin']):
                            st.session_state.selected_product = row['parent_asin']
                            st.switch_page("pages/1_Sentiment_Overview.py")

    # --- Pagination Buttons ---
    st.markdown("---")
    total_pages = (total_results + PRODUCTS_PER_PAGE - 1) // PRODUCTS_PER_PAGE
    if total_pages > 1:
        nav_cols = st.columns([1, 1, 1])
        with nav_cols[0]:
            if st.session_state.page > 0:
                if st.button("⬅️ Previous Page"):
                    st.session_state.page -= 1
                    st.rerun()
        with nav_cols[1]:
            st.write(f"Page {st.session_state.page + 1} of {total_pages}")
        with nav_cols[2]:
            if (st.session_state.page + 1) < total_pages:
                if st.button("Next Page ➡️"):
                    st.session_state.page += 1
                    st.rerun()
