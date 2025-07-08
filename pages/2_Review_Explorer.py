# pages/2_Review_Explorer.py
import streamlit as st
from utils.database_utils import (
    connect_to_db,
    get_product_details,
    get_product_date_range,
    get_paginated_reviews
)

# --- Page Configuration and Constants ---
st.set_page_config(layout="wide", page_title="Review Explorer")
DB_PATH = "amazon_reviews_top100.duckdb"
REVIEWS_PER_PAGE = 10
conn = connect_to_db(DB_PATH)

# --- Main App Logic ---
def main():
    st.title("📝 Review Explorer")

    if st.button("⬅️ Back to Sentiment Overview"):
        st.switch_page("pages/1_Sentiment_Overview.py")

    # --- Check for Selected Product ---
    if 'selected_product' not in st.session_state or st.session_state.selected_product is None:
        st.warning("Please select a product from the main search page first.")
        st.stop()
    selected_asin = st.session_state.selected_product

    product_details_df = get_product_details(conn, selected_asin)
    if product_details_df.empty:
        st.error("Could not find details for the selected product.")
        st.stop()
    product_details = product_details_df.iloc[0]
    st.header(product_details['product_title'])
    st.caption("Browse, filter, and sort all reviews for this product.")

    # --- Sidebar Filters (DEFINITIVE FIX) ---
    st.sidebar.header("📊 Interactive Filters")
    min_date_db, max_date_db = get_product_date_range(conn, selected_asin)
    
    # Define defaults
    default_date_range = (min_date_db, max_date_db)
    default_ratings = [1, 2, 3, 4, 5]
    default_sentiments = ['Positive', 'Negative', 'Neutral']

    # Initialize session state for filters if they don't exist
    if 'date_filter_explorer' not in st.session_state:
        st.session_state.date_filter_explorer = default_date_range
    if 'rating_filter_explorer' not in st.session_state:
        st.session_state.rating_filter_explorer = default_ratings
    if 'sentiment_filter_explorer' not in st.session_state:
        st.session_state.sentiment_filter_explorer = default_sentiments
    if 'review_page' not in st.session_state:
        st.session_state.review_page = 0
        
    # Callback to reset the page number when a filter changes
    def reset_page_number():
        st.session_state.review_page = 0

    # Callback to reset all filters to their default values
    def reset_all_filters():
        st.session_state.date_filter_explorer = default_date_range
        st.session_state.rating_filter_explorer = default_ratings
        st.session_state.sentiment_filter_explorer = default_sentiments
        st.session_state.review_page = 0

    # Create widgets. The `key` parameter links them to the session state.
    # We no longer need the 'value' or 'default' parameters.
    st.sidebar.date_input("Filter by Date Range", min_value=min_date_db, max_value=max_date_db, key='date_filter_explorer', on_change=reset_page_number)
    st.sidebar.multiselect("Filter by Star Rating", options=default_ratings, key='rating_filter_explorer', on_change=reset_page_number)
    st.sidebar.multiselect("Filter by Sentiment", options=default_sentiments, key='sentiment_filter_explorer', on_change=reset_page_number)
    
    st.sidebar.button("Reset All Filters", on_click=reset_all_filters, use_container_width=True, key='reset_button_explorer')

    # --- Sorting Control ---
    st.markdown("---")
    sort_by = st.selectbox(
        "Sort reviews by:",
        ("Newest First", "Oldest First", "Highest Rating", "Lowest Rating", "Most Helpful"),
        on_change=reset_page_number
    )

    # --- Data Fetching (reads from session state) ---
    reviews_df, total_reviews = get_paginated_reviews(
        _conn=conn,
        asin=selected_asin,
        date_range=st.session_state.date_filter_explorer,
        rating_filter=tuple(st.session_state.rating_filter_explorer),
        sentiment_filter=tuple(st.session_state.sentiment_filter_explorer),
        sort_by=sort_by,
        limit=REVIEWS_PER_PAGE,
        offset=st.session_state.review_page * REVIEWS_PER_PAGE
    )

    # --- Display Reviews and Pagination ---
    st.markdown("---")
    
    if reviews_df.empty:
        st.warning("No reviews match your current filter and sort criteria.")
    else:
        total_pages = (total_reviews + REVIEWS_PER_PAGE - 1) // REVIEWS_PER_PAGE
        st.info(f"Showing **{len(reviews_df)}** of **{total_reviews}** matching reviews. (Page **{st.session_state.review_page + 1}** of **{total_pages}**)")

        for _, review in reviews_df.iterrows():
            with st.container(border=True):
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.subheader(review['review_title'])
                    st.caption(f"Reviewed on: {review['date']} | Sentiment: {review['sentiment']}")
                    st.markdown(f"> {review['text']}")
                with col2:
                    st.metric("⭐ Rating", f"{review['rating']:.1f}")
                    st.metric("👍 Helpful Votes", f"{review['helpful_vote']}")
        
        st.markdown("---")
        nav_cols = st.columns([1, 1, 1])
        with nav_cols[0]:
            if st.session_state.review_page > 0:
                if st.button("⬅️ Previous Page"):
                    st.session_state.review_page -= 1
                    st.rerun()
        with nav_cols[1]:
            st.write(f"Page {st.session_state.review_page + 1} of {total_pages}")
        with nav_cols[2]:
            if (st.session_state.review_page + 1) < total_pages:
                if st.button("Next Page ➡️"):
                    st.session_state.review_page += 1
                    st.rerun()

if __name__ == "__main__":
    main()
