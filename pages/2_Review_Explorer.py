# pages/2_Review_Explorer.py
import streamlit as st
import re
import numpy as np
import plotly.express as px
from streamlit_plotly_events import plotly_events
from utils.database_utils import (
    connect_to_db,
    get_product_details,
    get_product_date_range,
    get_paginated_reviews,
    get_single_review_details,
    get_all_filtered_reviews 
)

# --- Page Configuration and Constants ---
st.set_page_config(layout="wide", page_title="Review Explorer")
DB_PATH = "amazon_reviews_final.duckdb"
#REVIEWS_PER_PAGE = 10
conn = connect_to_db(DB_PATH)

# --- Helper function to convert DataFrame to CSV ---
@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

# --- Main App Logic ---
def main():

    if st.button("⬅️ Back to Sentiment Overview"):
        st.switch_page("pages/1_Sentiment_Overview.py")
    title_col, help_col = st.columns([10, 1])
    with title_col:
        st.markdown("# 📝 Review Explorer") # Use markdown for H1 title
    with help_col:
        with st.popover("ⓘ"):
            st.markdown("##### What is this page for?")
            st.markdown("This page allows you to dive into the individual reviews. You can search for specific keywords, apply detailed filters, and sort the results to find the exact feedback you're looking for.")
            st.markdown("##### How do I use it?")
            st.markdown("Use the **sidebar** to filter reviews by date, rating, sentiment, or purchase status. Use the **search bar** below to find reviews containing a specific word or phrase.")
            st.markdown("##### Pro Tip:")
            st.markdown("Scroll down to the 'Advanced Analysis' section to find reviews where the star rating and the written sentiment don't match up!")
            
    # --- Check for Selected Product ---
    if 'selected_product' not in st.session_state or st.session_state.selected_product is None:
        st.warning("Please select a product from the main search page first.")
        st.stop()
    selected_asin = st.session_state.selected_product

    product_details = get_product_details(conn, selected_asin).iloc[0]
 
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
    if 'reviews_per_page' not in st.session_state: st.session_state.reviews_per_page = 10
        
    def reset_page_number():
        st.session_state.review_page = 0
    def reset_all_explorer_filters():
        st.session_state.explorer_date_filter = default_date_range
        st.session_state.explorer_rating_filter = default_ratings
        st.session_state.explorer_sentiment_filter = default_sentiments
        st.session_state.explorer_verified_filter = default_verified
        st.session_state.explorer_search_term = ""
        st.session_state.review_page = 0
    
    st.sidebar.date_input("Filter by Date Range", key='explorer_date_filter', on_change=reset_page_number)
    st.sidebar.multiselect("Filter by Star Rating", options=[1, 2, 3, 4, 5], key='explorer_rating_filter', on_change=reset_page_number)
    st.sidebar.multiselect("Filter by Sentiment", options=['Positive', 'Negative', 'Neutral'], key='explorer_sentiment_filter', on_change=reset_page_number)
    st.sidebar.radio("Filter by Purchase Status", ["All", "Verified Only", "Not Verified"], key='explorer_verified_filter', on_change=reset_page_number)
    st.sidebar.button("Reset All Filters", on_click=reset_all_explorer_filters, use_container_width=True)
    # --- Controls for Sorting and Searching (WITH CONDITIONAL LOGIC) ---
    c1, c2 = st.columns([1, 1])
    
    with c2:
        st.session_state.explorer_search_term = st.text_input(
            "Search within review text:", 
            value=st.session_state.explorer_search_term,
            on_change=reset_page_number,
            placeholder="e.g., battery life, easy to use"
        )
    
    with c1:
        sort_options = ("Newest First", "Oldest First", "Highest Rating", "Lowest Rating", "Most Helpful")
        search_active = bool(st.session_state.explorer_search_term)
        
        # If search is active, force sort to "Most Helpful". Otherwise, use the user's selection.
        if search_active:
            current_index = sort_options.index("Most Helpful")
        else:
            # Check if a sort option has been selected before, otherwise default to "Newest First"
            current_index = st.session_state.get('sort_by_index', 0)

        # The selectbox is disabled when a search term is entered
        sort_by = st.selectbox(
            "Sort reviews by:",
            sort_options,
            index=current_index,
            on_change=lambda: st.session_state.update(sort_by_index=sort_options.index(st.session_state.sort_selector)),
            disabled=search_active,
            key='sort_selector'
        )

    # --- Data Fetching ---
    paginated_reviews_df, total_reviews = get_paginated_reviews(
        _conn=conn,
        asin=selected_asin,
        date_range=st.session_state.explorer_date_filter,
        rating_filter=tuple(st.session_state.explorer_rating_filter),
        sentiment_filter=tuple(st.session_state.explorer_sentiment_filter),
        verified_filter=st.session_state.explorer_verified_filter,
        search_term=st.session_state.explorer_search_term,
        sort_by=sort_by,
        limit=st.session_state.reviews_per_page,
        offset=st.session_state.review_page * st.session_state.reviews_per_page
    )
    
    # --- Display Results and Export Button ---
    st.markdown("---")
    
    if paginated_reviews_df.empty:
        st.warning("No reviews match your current filter and search criteria.")
    else:
        # Display informational message and export button
        info_c1, page_c3, export_c2 = st.columns([2, 1, 1])
        with info_c1:
            total_pages = (total_reviews + st.session_state.reviews_per_page - 1) // st.session_state.reviews_per_page
            st.info(f"Showing **{len(paginated_reviews_df)}** of **{total_reviews}** matching reviews. (Page **{st.session_state.review_page + 1}** of **{total_pages}**)")
        with page_c3:
            st.selectbox(
                "Reviews per page:",
                options=[10, 25, 50],
                key='reviews_per_page',
                on_change=reset_page_number
            )
        with export_c2:
            st.download_button(
               label="📥 Export to CSV",
               data=convert_df_to_csv(get_all_filtered_reviews(
                _conn=conn, asin=selected_asin, date_range=st.session_state.explorer_date_filter,
                rating_filter=tuple(st.session_state.explorer_rating_filter), sentiment_filter=tuple(st.session_state.explorer_sentiment_filter),
                verified_filter=st.session_state.explorer_verified_filter, search_term=st.session_state.explorer_search_term
            )),
               file_name=f"{selected_asin}_reviews.csv",
               mime="text/csv",
               use_container_width=True,
               help="Download all reviews that match the current filter settings."
            )

        # Loop to display reviews with keyword highlighting
        for _, review in paginated_reviews_df.iterrows():
            with st.container(border=True):
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.subheader(review['review_title'])
                    caption_parts = ["✅ Verified" if review['verified_purchase'] else "❌ Not Verified", f"Reviewed on: {review['date']}", f"Sentiment: {review['sentiment']}"]
                    st.caption(" | ".join(caption_parts))
                    
                    search_term = st.session_state.explorer_search_term
                    review_text = review['text']
                    
                    if search_term:
                        highlighted_text = re.sub(f'({re.escape(search_term)})', r'<mark>\1</mark>', review_text, flags=re.IGNORECASE)
                        st.markdown(f"> {highlighted_text}", unsafe_allow_html=True)
                    else:
                        st.markdown(f"> {review_text}")
                with col2:
                    st.metric("⭐ Rating", f"{review['rating']:.1f}")
                    st.metric("👍 Helpful", f"{review['helpful_vote']}")
        
        # --- Pagination Buttons ---
        st.markdown("---")
        if total_reviews > st.session_state.reviews_per_page:
            nav_cols = st.columns([1, 1, 1])
            if st.session_state.review_page > 0:
                nav_cols[0].button("⬅️ Previous Page", on_click=lambda: setattr(st.session_state, 'review_page', st.session_state.review_page - 1), use_container_width=True)
            nav_cols[1].write(f"Page {st.session_state.review_page + 1} of {total_pages}")
            if (st.session_state.review_page + 1) < total_pages:
                nav_cols[2].button("Next Page ➡️", on_click=lambda: setattr(st.session_state, 'review_page', st.session_state.review_page + 1), use_container_width=True)

     # --- MOVED: RATING VS TEXT DISCREPANCY PLOT ---
    st.markdown("---")

    with st.expander("🔬 Advanced Analysis: Find Mismatched Reviews"):
        st.markdown("##### What is this?")
        st.markdown(
            "This scatter plot helps you find reviews where the **star rating might not match the sentiment of the written text**. For example, a 5-star review with negative language ('The product is great, *but the battery life is terrible*') or a 1-star review with positive language ('I hated it, *but the design was beautiful*')."
        )
        st.markdown("##### How do I use it?")
        st.markdown(
            "Each dot is a review. **Click on any dot** to see the full review text on the right. The color indicates the 'Discrepancy Score'—brighter dots have a bigger mismatch between their rating and text sentiment."
        )
        all_filtered_df_for_plot = get_all_filtered_reviews(
        _conn=conn, asin=selected_asin, date_range=st.session_state.explorer_date_filter,
        rating_filter=tuple(st.session_state.explorer_rating_filter), sentiment_filter=tuple(st.session_state.explorer_sentiment_filter),
        verified_filter=st.session_state.explorer_verified_filter, search_term=st.session_state.explorer_search_term
        )
        if all_filtered_df_for_plot.empty:
            st.warning("No review data available for the selected filters to generate this plot.")
        else:
            # Add necessary columns for plotting
            rng = np.random.default_rng(seed=42)
            all_filtered_df_for_plot['rating_jittered'] = all_filtered_df_for_plot['rating'] + rng.uniform(-0.1, 0.1, size=len(all_filtered_df_for_plot))
            all_filtered_df_for_plot['text_polarity_jittered'] = all_filtered_df_for_plot['sentiment_score'] + rng.uniform(-0.02, 0.02, size=len(all_filtered_df_for_plot))
            all_filtered_df_for_plot['text_polarity'] = all_filtered_df_for_plot['sentiment_score']
            all_filtered_df_for_plot['discrepancy'] = (all_filtered_df_for_plot['text_polarity'] - ((all_filtered_df_for_plot['rating'] - 3) / 2.0)).abs()
    
            plot_col, review_col = st.columns([2, 1])
            with plot_col:
                fig = px.scatter(
                    all_filtered_df_for_plot, x="rating_jittered", y="text_polarity_jittered", color="discrepancy",
                    color_continuous_scale=px.colors.sequential.Viridis,
                    labels={"rating_jittered": "Star Rating", "text_polarity_jittered": "Sentiment Score", "discrepancy": "Discrepancy Score"},
                    hover_name="review_title",
                    hover_data={"rating": True, "sentiment": True, "discrepancy": ":.2f", "rating_jittered": False, "text_polarity_jittered": False}
                )
                fig.update_layout(clickmode='event+select')
                fig.update_traces(marker_size=10)
                selected_points = plotly_events(fig, click_event=True, key="plotly_event_selector")
                if selected_points and 'pointIndex' in selected_points[0]:
                    point_index = selected_points[0]['pointIndex']
                    if point_index < len(all_filtered_df_for_plot):
                        clicked_id = all_filtered_df_for_plot.iloc[point_index]['review_id']
                        if st.session_state.selected_review_id != clicked_id:
                            st.session_state.selected_review_id = clicked_id
                            st.rerun()
            with review_col:
                if st.session_state.selected_review_id:
                    if st.session_state.selected_review_id in all_filtered_df_for_plot['review_id'].values:
                        st.markdown("#### Selected Review Details")
                        review_details = get_single_review_details(conn, st.session_state.selected_review_id)
                        if review_details is not None:
                            st.subheader(review_details.get('review_title', 'No Title'))
                            caption_parts = [f"Reviewed on: {review_details.get('date', 'N/A')}", f"👍 {int(review_details.get('helpful_vote', 0))} helpful votes"]
                            st.caption(" | ".join(caption_parts))
                            st.markdown(f"> {review_details.get('text', 'Review text not available.')}")
                        if st.button("Close Review", key="close_review_button"):
                            st.session_state.selected_review_id = None
                            st.rerun()
                            
if __name__ == "__main__":
    main()
