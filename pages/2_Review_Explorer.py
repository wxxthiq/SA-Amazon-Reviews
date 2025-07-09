# pages/2_Review_Explorer.py
import streamlit as st
from utils.database_utils import (
    connect_to_db,
    get_product_details,
    get_product_date_range,
    get_paginated_reviews
)
import re # --- NEW: Import the regular expression module

# --- Page Configuration and Constants ---
st.set_page_config(layout="wide", page_title="Review Explorer")
DB_PATH = "amazon_reviews_top100.duckdb"
REVIEWS_PER_PAGE = 10
conn = connect_to_db(DB_PATH)

# --- Helper function to convert DataFrame to CSV ---
@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

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

    product_details = get_product_details(conn, selected_asin).iloc[0]
    st.header(product_details['product_title'])
    st.caption("Browse, filter, and sort all reviews for this product. Use the search box or export your filtered results.")

    # --- Sidebar Filters ---
    st.sidebar.header("📊 Interactive Filters")
    min_date_db, max_date_db = get_product_date_range(conn, selected_asin)
    
    # Initialize session state variables
    if 'explorer_date_filter' not in st.session_state: st.session_state.explorer_date_filter = (min_date_db, max_date_db)
    if 'explorer_rating_filter' not in st.session_state: st.session_state.explorer_rating_filter = [1, 2, 3, 4, 5]
    if 'explorer_sentiment_filter' not in st.session_state: st.session_state.explorer_sentiment_filter = ['Positive', 'Negative', 'Neutral']
    if 'explorer_verified_filter' not in st.session_state: st.session_state.explorer_verified_filter = "All"
    if 'review_page' not in st.session_state: st.session_state.review_page = 0
    if 'explorer_search_term' not in st.session_state: st.session_state.explorer_search_term = ""
        
    def reset_page_number():
        st.session_state.review_page = 0
    
    st.sidebar.date_input("Filter by Date Range", key='explorer_date_filter', on_change=reset_page_number)
    st.sidebar.multiselect("Filter by Star Rating", options=[1, 2, 3, 4, 5], key='explorer_rating_filter', on_change=reset_page_number)
    st.sidebar.multiselect("Filter by Sentiment", options=['Positive', 'Negative', 'Neutral'], key='explorer_sentiment_filter', on_change=reset_page_number)
    st.sidebar.radio("Filter by Purchase Status", ["All", "Verified Only", "Not Verified"], key='explorer_verified_filter', on_change=reset_page_number)

    # --- Controls for Sorting and Searching ---
    st.markdown("---")
    c1, c2 = st.columns([1, 1])
    with c1:
        sort_by = st.selectbox(
            "Sort reviews by:",
            ("Newest First", "Oldest First", "Highest Rating", "Lowest Rating", "Most Helpful"),
            index=0,
            on_change=reset_page_number
        )
    with c2:
        st.session_state.explorer_search_term = st.text_input(
            "Search within review text:", 
            value=st.session_state.explorer_search_term,
            on_change=reset_page_number,
            placeholder="e.g., battery life, easy to use"
        )

    # --- Data Fetching ---
    paginated_reviews_df, total_reviews, all_filtered_df = get_paginated_reviews(
        _conn=conn,
        asin=selected_asin,
        date_range=st.session_state.explorer_date_filter,
        rating_filter=tuple(st.session_state.explorer_rating_filter),
        sentiment_filter=tuple(st.session_state.explorer_sentiment_filter),
        verified_filter=st.session_state.explorer_verified_filter,
        search_term=st.session_state.explorer_search_term,
        sort_by=sort_by,
        limit=REVIEWS_PER_PAGE,
        offset=st.session_state.review_page * REVIEWS_PER_PAGE
    )

    # --- Display Results and Export Button ---
    st.markdown("---")
    
    if paginated_reviews_df.empty:
        st.warning("No reviews match your current filter and sort criteria.")
    else:
        info_c1, export_c2 = st.columns([3, 1])
        with info_c1:
            total_pages = (total_reviews + REVIEWS_PER_PAGE - 1) // REVIEWS_PER_PAGE
            st.info(f"Showing **{len(paginated_reviews_df)}** of **{total_reviews}** matching reviews. (Page **{st.session_state.review_page + 1}** of **{total_pages}**)")
        with export_c2:
            st.download_button(
               label="📥 Export to CSV",
               data=convert_df_to_csv(all_filtered_df),
               file_name=f"{selected_asin}_reviews.csv",
               mime="text/csv",
               use_container_width=True
            )

        for _, review in paginated_reviews_df.iterrows():
            with st.container(border=True):
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.subheader(review['review_title'])
                    caption_parts = ["✅ Verified" if review['verified_purchase'] else "❌ Not Verified", f"Reviewed on: {review['date']}", f"Sentiment: {review['sentiment']}"]
                    st.caption(" | ".join(caption_parts))

                    # --- UPDATED: Logic to highlight the search term ---
                    search_term = st.session_state.explorer_search_term
                    review_text = review['text']
                    
                    if search_term:
                        # Use re.sub for case-insensitive replacement and wrap with <mark> tags for highlighting
                        highlighted_text = re.sub(f'({re.escape(search_term)})', r'<mark>\1</mark>', review_text, flags=re.IGNORECASE)
                        st.markdown(f"> {highlighted_text}", unsafe_allow_html=True)
                    else:
                        st.markdown(f"> {review_text}")
                    # --- END of update ---

                with col2:
                    st.metric("⭐ Rating", f"{review['rating']:.1f}")
                    st.metric("👍 Helpful", f"{review['helpful_vote']}")
        
        # --- Pagination Buttons ---
        st.markdown("---")
        if total_reviews > REVIEWS_PER_PAGE:
            nav_cols = st.columns([1, 1, 1])
            if st.session_state.review_page > 0:
                nav_cols[0].button("⬅️ Previous Page", on_click=lambda: setattr(st.session_state, 'review_page', st.session_state.review_page - 1), use_container_width=True)
            nav_cols[1].write(f"Page {st.session_state.review_page + 1} of {total_pages}")
            if (st.session_state.review_page + 1) < total_pages:
                nav_cols[2].button("Next Page ➡️", on_click=lambda: setattr(st.session_state, 'review_page', st.session_state.review_page + 1), use_container_width=True)

if __name__ == "__main__":
    main()
