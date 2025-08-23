# app.py (Main Search Page)
import streamlit as st
import pandas as pd
from utils.database_utils import connect_to_db, get_all_categories, get_filtered_products, a_download_data_with_versioning

# --- App Configuration ---
DB_PATH = "amazon_reviews_final.duckdb"
DB_VERSION = 1
PLACEHOLDER_IMAGE_URL = "https://via.placeholder.com/200"
KAGGLE_DATASET_SLUG = "wathiqsoualhi/amazon-reviews-duckdb"


# --- Page Setup ---
st.set_page_config(layout="wide", page_title="Amazon Review Search")
st.title("🔎 Amazon Product Search")

# ADD THIS CSS BLOCK
st.markdown("""
    <style>
    .product-container {
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        height: 100%;
    }
    .product-image-container {
        height: 200px; /* Fixed height for the image container */
        display: flex;
        align-items: center;
        justify-content: center;
        overflow: hidden;
        margin-bottom: 10px;
    }
    .product-image-container img {
        max-height: 100%;
        max-width: 100%;
        object-fit: contain; /* Scales the image to fit */
    }
    </style>
""", unsafe_allow_html=True)

st.info("ℹ️ **How to Use:** Start by selecting a product category from the dropdown menu. You can then use the search bar and advanced filters to find specific products.")

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
if 'products_per_page' not in st.session_state: st.session_state.products_per_page = 8

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
    adv_col1, adv_col2, adv_col3 = st.columns(3) # Create a third column
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
            max_value=50000, 
            value=st.session_state.review_count_range,
            step=100
        )
    # ADD THIS NEW WIDGET
    with adv_col3:
        st.session_state.products_per_page = st.selectbox(
            "Products per Page:",
            options=[8, 16, 24, 32],
            help="Select how many products to show on each page."
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
        limit=st.session_state.products_per_page,
        offset=st.session_state.page * st.session_state.products_per_page
    )

    st.header(f"Found {total_results} Products in '{st.session_state.category}'")

    # ... (inside the `else` block after "Found {total_results} Products...")

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
                        # MODIFIED SECTION
                        image_urls_str = row.get('image_urls')
                        thumbnail_url = image_urls_str.split(',')[0] if pd.notna(image_urls_str) else PLACEHOLDER_IMAGE_URL

                        st.markdown(f"""
                            <div class="product-container">
                                <div>
                                    <div class="product-image-container">
                                        <img src="{thumbnail_url}" class="product-image">
                                    </div>
                                    <p><strong>{row['product_title']}</strong></p>
                                </div>
                                <div>
                        """, unsafe_allow_html=True)

                        avg_rating = row.get('average_rating', 0)
                        review_count = row.get('review_count', 0)
                        st.caption(f"Avg. Rating: {avg_rating:.2f} ⭐ ({int(review_count)} reviews)")

                        if st.button("View Details", key=row['parent_asin']):
                            st.session_state.selected_product = row['parent_asin']
                            st.switch_page("pages/1_Sentiment_Overview.py")

                        st.markdown("</div></div>", unsafe_allow_html=True) # Closes the divs

    # --- Pagination Buttons ---
    st.markdown("---")
    total_pages = (total_results + st.session_state.products_per_page - 1) // st.session_state.products_per_page
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
