# pages/2_Review_Explorer.py
import streamlit as st
from utils.database_utils import (
    connect_to_db,
    get_product_details,
    get_product_date_range,
    get_paginated_reviews # Import the new function
)

# --- Page Configuration and State Initialization ---
st.set_page_config(layout="wide", page_title="Review Explorer")

if 'review_page' not in st.session_state:
    st.session_state.review_page = 0

# --- Constants & DB Connection ---
DB_PATH = "amazon_reviews_top100.duckdb"
REVIEWS_PER_PAGE = 10
conn = connect_to_db(DB_PATH)

# --- Main App Logic ---
def main():
    st.title("📝 Review Explorer")

    # --- Check for Selected Product ---
    if 'selected_product' not in st.session_state or st.session_state.selected_product is None:
        st.warning("Please select a product from the main search page first.")
        if st.button("⬅️ Back to Search"):
            st.switch_page("app.py")
        st.stop()
    selected_asin = st.session_state.selected_product

    # --- Load Product Data ---
    product_details_df = get_product_details(conn, selected_asin)
    if product_details_df.empty:
        st.error("Could not find details for the selected product.")
        st.stop()
    product_details = product_details_df.iloc[0]
    st.header(product_details['product_title'])
    st.caption("Browse, filter, and sort all reviews for this product.")

    # --- Sidebar Filters ---
    st.sidebar.header("📊 Interactive Filters")
    
    def reset_page_number():
        st.session_state.review_page = 0

    min_date_db, max_date_db = get_product_date_range(conn, selected_asin)
    default_date_range = (min_date_db, max_date_db)
    default_ratings = [1, 2, 3, 4, 5]
    default_sentiments = ['Positive', 'Negative', 'Neutral']

    selected_date_range = st.sidebar.date_input("Filter by Date Range", value=default_date_range, min_value=min_date_db, max_value=max_date_db, on_change=reset_page_number)
    selected_ratings = st.sidebar.multiselect("Filter by Star Rating", options=default_ratings, default=default_ratings, on_change=reset_page_number)
    selected_sentiments = st.sidebar.multiselect("Filter by Sentiment", options=default_sentiments, default=default_sentiments, on_change=reset_page_number)

    # --- Sorting Control ---
    st.markdown("---")
    sort_by = st.selectbox(
        "Sort reviews by:",
        ("Newest First", "Oldest First", "Highest Rating", "Lowest Rating", "Most Helpful"),
        on_change=reset_page_number
    )

    # --- Data Fetching ---
    reviews_df, total_reviews = get_paginated_reviews(
        _conn=conn,
        asin=selected_asin,
        date_range=selected_date_range,
        rating_filter=tuple(selected_ratings),
        sentiment_filter=tuple(selected_sentiments),
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

        # Display each review
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
        
        # --- Pagination Buttons ---
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
