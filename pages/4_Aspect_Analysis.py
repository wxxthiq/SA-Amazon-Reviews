# pages/4_Aspect_Analysis.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import altair as alt
import spacy
from collections import Counter
import re
from textblob import TextBlob
import plotly.graph_objects as go
from utils.database_utils import (
    connect_to_db,
    get_product_details,
    get_reviews_for_product,
    get_product_date_range
)

# --- Page Configuration and NLP Model Loading ---
st.set_page_config(layout="wide", page_title="Aspect Analysis")

@st.cache_resource
def load_spacy_model():
    return spacy.load("en_core_web_sm")

@st.cache_data
def convert_df_to_csv(df):
    """Converts a dataframe to a downloadable CSV file."""
    return df.to_csv(index=False).encode('utf-8')

# --- Definitive Aspect Extraction Function ---
@st.cache_data
def extract_aspects_with_sentiment(dataf):
    """
    Uses a definitive, fully automated filtering approach to extract 
    high-quality aspects using spaCy's built-in stop word list.
    """
    aspect_sentiments = []
    stop_words = nlp.Defaults.stop_words
    for doc, sentiment in zip(nlp.pipe(dataf['text']), dataf['sentiment']):
        for chunk in doc.noun_chunks:
            tokens = [token for token in chunk]
            while len(tokens) > 0 and (tokens[0].is_stop or tokens[0].pos_ == 'DET'):
                tokens.pop(0)
            while len(tokens) > 0 and tokens[-1].is_stop:
                tokens.pop(-1)
            if not tokens:
                continue
            final_aspect = " ".join(token.lemma_.lower() for token in tokens)
            if len(final_aspect) > 3 and final_aspect not in stop_words:
                aspect_sentiments.append({'aspect': final_aspect, 'sentiment': sentiment})
    if not aspect_sentiments:
        return pd.DataFrame()
    return pd.DataFrame(aspect_sentiments)
nlp = load_spacy_model()
DB_PATH = "amazon_reviews_final.duckdb"
conn = connect_to_db(DB_PATH)
REVIEWS_PER_PAGE = 5

# --- Main App Logic ---
def main():
    st.title("🔎 Aspect-Based Sentiment Analysis")
    
    if 'aspect_review_page' not in st.session_state:
        st.session_state.aspect_review_page = 0
        
    if st.button("⬅️ Back to Sentiment Overview"):
        st.switch_page("pages/1_Sentiment_Overview.py")

    # --- Check for Selected Product ---
    if 'selected_product' not in st.session_state or st.session_state.selected_product is None:
        st.warning("Please select a product from the main search page first.")
        st.stop()
    selected_asin = st.session_state.selected_product

    # --- Load Product Data ---
    product_details = get_product_details(conn, selected_asin).iloc[0]
    st.header(product_details['product_title'])
    st.caption("This page automatically identifies key product features and analyzes the specific sentiment towards them.")

    # --- Sidebar Filters (COMPLETE SET) ---
    st.sidebar.header("🔬 Aspect Analysis Filters")
    min_date_db, max_date_db = get_product_date_range(conn, selected_asin)
    
    default_date_range = (min_date_db, max_date_db)
    default_ratings = [1, 2, 3, 4, 5]
    default_sentiments = ['Positive', 'Negative', 'Neutral'] # Added sentiment filter
    default_verified = "All"

    # Callback to reset all filters on this page
    def reset_all_aspect_filters():
        st.session_state.aspect_date_filter = default_date_range
        st.session_state.aspect_rating_filter = default_ratings
        st.session_state.aspect_sentiment_filter = default_sentiments
        st.session_state.aspect_verified_filter = default_verified
    
    selected_date_range = st.sidebar.date_input("Filter by Date Range", value=default_date_range, key='aspect_date_filter')
    selected_ratings = st.sidebar.multiselect("Filter by Star Rating", options=default_ratings, default=default_ratings, key='aspect_rating_filter')
    selected_sentiments = st.sidebar.multiselect("Filter by Sentiment", options=default_sentiments, default=default_sentiments, key='aspect_sentiment_filter') # Added widget
    selected_verified = st.sidebar.radio("Filter by Purchase Status", ["All", "Verified Only", "Not Verified"], index=0, key='aspect_verified_filter')
    
    # Add the reset button
    st.sidebar.button("Reset All Filters", on_click=reset_all_aspect_filters, use_container_width=True, key='reset_aspect_filters')
    # Load data based on all filters
    chart_data = get_reviews_for_product(conn, selected_asin, selected_date_range, tuple(selected_ratings), tuple(selected_sentiments), selected_verified)

    st.markdown("---")
    if chart_data.empty:
        st.warning("No review data available for the selected filters.")
        st.stop()
        
    st.info(f"Analyzing aspects from **{len(chart_data)}** reviews matching your criteria.")

    # --- FIX: Call the correct function to create aspect_df ---
    aspect_df = extract_aspects_with_sentiment(chart_data)

    if aspect_df.empty:
        st.warning("No distinct aspects could be extracted from the filtered reviews.")
        st.stop()
    # --- Aspect Summary Chart (with Interactive Sorting) ---
    st.markdown("### Aspect Analysis")
    
    # Create a two-column layout
    col1, col2 = st.columns([3, 2]) # Give more space to the main chart
    
    with col1:
        st.markdown("#### Aspect Sentiment Summary")
        sort_option = st.selectbox(
            "Sort aspects by:",
            ("Most Discussed", "Most Positive", "Most Negative", "Most Controversial"),
            key="aspect_sort_selector"
        )
        num_aspects_to_show = st.slider(
            "Select number of top aspects to display:",
            min_value=3, max_value=20, value=10, key="detailed_aspect_slider"
        )
    
        # --- Data Processing and Sorting Logic (Unchanged) ---
        sentiment_counts = aspect_df.groupby(['aspect', 'sentiment']).size().reset_index(name='count')
        pivot_df = sentiment_counts.pivot_table(index='aspect', columns='sentiment', values='count', fill_value=0)
        for col in ['Positive', 'Neutral', 'Negative']:
            if col not in pivot_df.columns: pivot_df[col] = 0
        pivot_df['total'] = pivot_df['Positive'] + pivot_df['Neutral'] + pivot_df['Negative']
        pivot_df['positive_pct'] = pivot_df['Positive'] / pivot_df['total']
        pivot_df['negative_pct'] = pivot_df['Negative'] / pivot_df['total']
        pivot_df['controversy'] = pivot_df['positive_pct'] * pivot_df['negative_pct']
    
        if sort_option == "Most Positive": sort_field, sort_order = 'positive_pct', 'descending'
        elif sort_option == "Most Negative": sort_field, sort_order = 'negative_pct', 'descending'
        elif sort_option == "Most Controversial": sort_field, sort_order = 'controversy', 'descending'
        else: sort_field, sort_order = 'total', 'descending'
            
        top_aspects_sorted = pivot_df.nlargest(num_aspects_to_show, sort_field).index.tolist()
        top_aspects_df = sentiment_counts[sentiment_counts['aspect'].isin(top_aspects_sorted)]
        
        # --- Create the Summary Chart (Unchanged) ---
        summary_chart = alt.Chart(top_aspects_df).mark_bar().encode(
            y=alt.Y('aspect:N', title='Product Aspect', sort=alt.EncodingSortField(field=sort_field, op="sum", order=sort_order)),
            x=alt.X('sum(count):Q', stack="normalize", title="Sentiment Distribution", axis=alt.Axis(format='%')),
            color=alt.Color('sentiment:N', scale=alt.Scale(domain=['Positive', 'Neutral', 'Negative'], range=['#1a9850', '#cccccc', '#d73027']), legend=alt.Legend(title="Sentiment")),
            tooltip=[alt.Tooltip('aspect', title='Aspect'), alt.Tooltip('sentiment', title='Sentiment'), alt.Tooltip('sum(count):Q', title='Review Count')]
        ).properties(title=f"Sentiment Analysis of Top {num_aspects_to_show} Aspects")
        st.altair_chart(summary_chart, use_container_width=True)
    
    with col2:
        st.markdown("#### Comparative Analysis")
        st.caption("Select 2+ aspects to compare their sentiment profiles.")
        
        if top_aspects_sorted:
            selected_for_comparison = st.multiselect(
                "Select aspects to compare:",
                options=top_aspects_sorted,
                default=top_aspects_sorted[:3] if len(top_aspects_sorted) >= 3 else top_aspects_sorted
            )
    
            if len(selected_for_comparison) >= 2:
                comparison_df = sentiment_counts[sentiment_counts['aspect'].isin(selected_for_comparison)]
                radar_df = comparison_df.pivot_table(index='aspect', columns='sentiment', values='count', fill_value=0)
                
                categories = ['Positive', 'Negative', 'Neutral']
                for sent in categories:
                    if sent not in radar_df.columns: radar_df[sent] = 0
                radar_df = radar_df[categories]
    
                fig = go.Figure()
                for aspect in radar_df.index:
                    hover_text = [f"{count} {cat} mentions" for cat, count in zip(categories, radar_df.loc[aspect].values)]
                    fig.add_trace(go.Scatterpolar(
                        r=radar_df.loc[aspect].values,
                        theta=categories,
                        fill='toself',
                        name=aspect,
                        hoverinfo='name+text',
                        text=hover_text
                    ))
                
                # --- FIX: Simplify the layout for a cleaner look ---
                fig.update_layout(
                  polar=dict(radialaxis=dict(visible=False)), # Hide the radial axis grid
                  showlegend=True,
                  title="Sentiment Profile Comparison"
                )
                st.plotly_chart(fig, use_container_width=True)

    # --- Interactive Aspect Explorer (ENHANCED) ---
    st.markdown("---")
    st.markdown("### 🔬 Interactive Aspect Explorer")
    # Callback to reset the page number for the reviews
    def reset_aspect_page_number():
        st.session_state.aspect_review_page = 0
    
    selected_aspect = st.selectbox(
        "Select an auto-detected aspect to analyze in detail:",
        options=["--- Select an Aspect ---"] + top_aspects_sorted,
        on_change=reset_aspect_page_number # Reset pagination if aspect changes
    )
    
    if selected_aspect != "--- Select an Aspect ---":
        aspect_df = chart_data[chart_data['text'].str.contains(r'\b' + re.escape(selected_aspect) + r'\b', case=False, na=False)].copy()
        
        st.markdown(f"---")
        st.markdown(f"#### Analysis for aspect: `{selected_aspect}` ({len(aspect_df)} mentions)")
        
        if aspect_df.empty:
            st.warning(f"No mentions of '{selected_aspect}' found with the current filters.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Rating Distribution for this Aspect**")
                # Prepare data and calculate percentages
                rating_counts_df = aspect_df['rating'].value_counts().reindex(range(1, 6), fill_value=0).reset_index()
                rating_counts_df.columns = ['rating', 'count']
                rating_counts_df['percentage'] = (rating_counts_df['count'] / len(aspect_df)) * 100
                rating_counts_df['rating_str'] = rating_counts_df['rating'].astype(str) + ' ⭐'
        
                # Base bar chart
                rating_bar_chart = alt.Chart(rating_counts_df).mark_bar().encode(
                    x=alt.X('count:Q', title='Number of Reviews'),
                    y=alt.Y('rating_str:N', sort=alt.EncodingSortField(field="rating", order="descending"), title=None),
                    color=alt.Color('rating:O',
                                    scale=alt.Scale(domain=[5, 4, 3, 2, 1], range=['#2ca02c', '#98df8a', '#ffdd71', '#ff9896', '#d62728']),
                                    legend=None),
                    tooltip=[alt.Tooltip('rating_str', title='Rating'), alt.Tooltip('count'), alt.Tooltip('percentage', format='.1f')]
                )
                # Text labels
                rating_text_labels = rating_bar_chart.mark_text(align='left', baseline='middle', dx=3, color='white').encode(
                    text=alt.Text('percentage:Q', format='.1f')
                )
                st.altair_chart(rating_bar_chart + rating_text_labels, use_container_width=True)
        
            with col2:
                st.markdown("**Sentiment Distribution for this Aspect**")
                # Prepare data and calculate percentages
                sentiment_counts_df = aspect_df['sentiment'].value_counts().reindex(['Positive', 'Neutral', 'Negative'], fill_value=0).reset_index()
                sentiment_counts_df.columns = ['sentiment', 'count']
                sentiment_counts_df['percentage'] = (sentiment_counts_df['count'] / len(aspect_df)) * 100
        
                # Base bar chart
                sentiment_bar_chart = alt.Chart(sentiment_counts_df).mark_bar().encode(
                    x=alt.X('count:Q', title='Number of Reviews'),
                    y=alt.Y('sentiment:N', sort=['Positive', 'Neutral', 'Negative'], title=None),
                    color=alt.Color('sentiment:N',
                                    scale=alt.Scale(domain=['Positive', 'Neutral', 'Negative'], range=['#1a9850', '#cccccc', '#d73027']),
                                    legend=None),
                    tooltip=[alt.Tooltip('sentiment'), alt.Tooltip('count'), alt.Tooltip('percentage', format='.1f')]
                )
                # Text labels
                sentiment_text_labels = sentiment_bar_chart.mark_text(align='left', baseline='middle', dx=3, color='white').encode(
                    text=alt.Text('percentage:Q', format='.1f')
                )
                st.altair_chart(sentiment_bar_chart + sentiment_text_labels, use_container_width=True)
            st.markdown("---")
            st.markdown("**Trends for this Aspect Over Time**")
            time_granularity = st.radio(
            "Select time period:",
            ("Monthly", "Weekly", "Daily"),
            index=0,
            horizontal=True,
            key="aspect_time_granularity"
            )
            
            time_df = aspect_df.copy()
            time_df['date'] = pd.to_datetime(time_df['date'])
            
            if time_granularity == 'Daily':
                time_df['period'] = time_df['date'].dt.date
            elif time_granularity == 'Weekly':
                time_df['period'] = time_df['date'].dt.to_period('W').dt.start_time
            else: # Monthly
                time_df['period'] = time_df['date'].dt.to_period('M').dt.start_time
                
            rating_counts_over_time = time_df.groupby(['period', 'rating']).size().reset_index(name='count')
            sentiment_counts_over_time = time_df.groupby(['period', 'sentiment']).size().reset_index(name='count')
        
            tab1, tab2 = st.tabs(["📈 Line Chart View", "📊 Area Chart View"])
        
            with tab1:
                col1, col2 = st.columns(2)
                with col1:
                    show_rating_trend = st.toggle('Show Average Rating Trend', key='aspect_rating_trend_toggle')
                    if not rating_counts_over_time.empty:
                        fig = px.line(
                            rating_counts_over_time, x='period', y='count', color='rating',
                            labels={'period': 'Date', 'count': 'Number of Reviews', 'rating': 'Star Rating'},
                            color_discrete_map={5: '#1a9850', 4: '#91cf60', 3: '#d9ef8b', 2: 'orange', 1: '#d73027'},
                            category_orders={"rating": [5, 4, 3, 2, 1]}
                        )
                        if show_rating_trend:
                            avg_rating_trend = time_df.groupby('period')['rating'].mean().reset_index()
                            fig.add_trace(go.Scatter(x=avg_rating_trend['period'], y=avg_rating_trend['rating'], mode='lines', name='Average Rating', yaxis='y2', line=dict(dash='dash', color='blue')))
                            fig.update_layout(title_text="Trend and Average Rating", yaxis2=dict(title='Avg Rating', overlaying='y', side='right', range=[1, 5]))
                        else:
                            fig.update_layout(title_text="Volume by Rating")
                        fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                        st.plotly_chart(fig, use_container_width=True)
        
                with col2:
                    show_sentiment_trend = st.toggle('Show Average Sentiment Trend', key='aspect_sentiment_trend_toggle')
                    if not sentiment_counts_over_time.empty:
                        fig = px.line(
                            sentiment_counts_over_time, x='period', y='count', color='sentiment',
                            labels={'period': 'Date', 'count': 'Number of Reviews', 'sentiment': 'Sentiment'},
                            color_discrete_map={'Positive': '#1a9850', 'Neutral': '#cccccc', 'Negative': '#d73027'},
                            category_orders={"sentiment": ["Positive", "Neutral", "Negative"]}
                        )
                        if show_sentiment_trend:
                            avg_sentiment_trend = time_df.groupby('period')['sentiment_score'].mean().reset_index()
                            fig.add_trace(go.Scatter(x=avg_sentiment_trend['period'], y=avg_sentiment_trend['sentiment_score'], mode='lines', name='Avg. Sentiment', yaxis='y2', line=dict(dash='dash', color='blue')))
                            fig.update_layout(title_text="Trend and Average Sentiment", yaxis2=dict(title='Avg Sentiment', overlaying='y', side='right', range=[-1, 1]))
                        else:
                            fig.update_layout(title_text="Volume by Sentiment")
                        fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                        st.plotly_chart(fig, use_container_width=True)
        
            with tab2:
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("#### Rating Distribution")
                    if not rating_counts_over_time.empty:
                        fig_area_rating = px.area(
                            rating_counts_over_time, x='period', y='count', color='rating', title="Distribution by Rating",
                            labels={'period': 'Date', 'count': 'Number of Reviews', 'rating': 'Star Rating'},
                            color_discrete_map={5: '#1a9850', 4: '#91cf60', 3: '#d9ef8b', 2: 'orange', 1: '#d73027'},
                            category_orders={"rating": [5, 4, 3, 2, 1]}
                        )
                        st.plotly_chart(fig_area_rating, use_container_width=True)
        
                with col2:
                    st.markdown("#### Sentiment Distribution")
                    if not sentiment_counts_over_time.empty:
                        fig_area_sentiment = px.area(
                            sentiment_counts_over_time, x='period', y='count', color='sentiment', title="Distribution by Sentiment",
                            labels={'period': 'Date', 'count': 'Number of Reviews', 'sentiment': 'Sentiment'},
                            color_discrete_map={'Positive': '#1a9850', 'Neutral': '#cccccc', 'Negative': '#d73027'},
                            category_orders={"sentiment": ["Positive", "Neutral", "Negative"]}
                        )
                        st.plotly_chart(fig_area_sentiment, use_container_width=True)
    
            # --- Example Reviews Display with Sorting and Pagination ---
            st.markdown("---")
            st.markdown("**Example Reviews Mentioning this Aspect**")
            # --- Create columns for sorting and downloading ---
            sort_col, download_col = st.columns([2, 1]) # Give more space to the sort dropdown

            with sort_col:
                sort_reviews_by = st.selectbox(
                    "Sort examples by:",
                    ("Most Helpful", "Newest", "Oldest", "Highest Rating", "Lowest Rating"),
                    key="aspect_review_sort",
                    on_change=reset_aspect_page_number
                )
            
            with download_col:
                if not aspect_df.empty:
                    csv_data = convert_df_to_csv(aspect_df)
                    st.download_button(
                       label="📥 Download Reviews",
                       data=csv_data,
                       file_name=f"{selected_asin}_{selected_aspect}_reviews.csv",
                       mime="text/csv",
                       use_container_width=True,
                       help=f"Download all {len(aspect_df)} reviews that mention '{selected_aspect}'"
                    )
    
            if sort_reviews_by == "Most Helpful":
                sorted_aspect_df = aspect_df.sort_values(by="helpful_vote", ascending=False)
            elif sort_reviews_by == "Highest Rating":
                sorted_aspect_df = aspect_df.sort_values(by=["rating", "helpful_vote"], ascending=[False, False])
            elif sort_reviews_by == "Lowest Rating":
                sorted_aspect_df = aspect_df.sort_values(by=["rating", "helpful_vote"], ascending=[True, False])
            elif sort_reviews_by == "Oldest":
                sorted_aspect_df = aspect_df.sort_values(by="date", ascending=True)
            else: # Newest
                sorted_aspect_df = aspect_df.sort_values(by="date", ascending=False)
    
            start_idx = st.session_state.aspect_review_page * REVIEWS_PER_PAGE
            end_idx = start_idx + REVIEWS_PER_PAGE
            reviews_to_display = sorted_aspect_df.iloc[start_idx:end_idx]
            
            # Helper function to highlight text
            def highlight_text(text, aspect):
                return re.sub(r'(\b' + re.escape(aspect) + r'\b)', r'**<span style="color:orange">\1</span>**', text, flags=re.IGNORECASE)
    
            for _, review in reviews_to_display.iterrows():
                with st.container(border=True):
                    st.subheader(review['review_title'])
                    caption_parts = []
                    if review['verified_purchase']:
                        caption_parts.append("✅ Verified")
                    caption_parts.append(f"Reviewed on: {review['date']}")
                    caption_parts.append(f"Rating: {review['rating']} ⭐")
                    caption_parts.append(f"Helpful Votes: {review['helpful_vote']} 👍")
                    st.caption(" | ".join(caption_parts))
                    
                    highlighted_review = highlight_text(review['text'], selected_aspect)
                    st.markdown(f"> {highlighted_review}", unsafe_allow_html=True)
            
            # Pagination Buttons
            total_reviews = len(sorted_aspect_df)
            total_pages = (total_reviews + REVIEWS_PER_PAGE - 1) // REVIEWS_PER_PAGE
            
            if total_pages > 1:
                st.caption(f"Page {st.session_state.aspect_review_page + 1} of {total_pages}")
                p_col1, p_col2 = st.columns(2)
                with p_col1:
                    if st.session_state.aspect_review_page > 0:
                        if st.button("⬅️ Previous 5 Reviews"):
                            st.session_state.aspect_review_page -= 1
                            st.rerun()
                with p_col2:
                    if end_idx < total_reviews:
                        if st.button("Next 5 Reviews ➡️"):
                            st.session_state.aspect_review_page += 1
                            st.rerun()

    # --- Comparative Aspect Analysis using Radar Chart ---
    st.markdown("---")
    st.markdown("### ⚖️ Comparative Aspect Analysis")
    st.caption("Select two or more aspects to compare their sentiment profiles. Hover over the points for details.")

    if top_aspects_sorted:
        selected_for_comparison = st.multiselect(
            "Select aspects to compare:",
            options=top_aspects_sorted,
            default=top_aspects_sorted[:3] if len(top_aspects_sorted) >= 3 else sorted
        )

        if len(selected_for_comparison) >= 2:
            comparison_df = sentiment_counts[sentiment_counts['aspect'].isin(selected_for_comparison)] # <--- CORRECTED VARIABLE
            
            radar_df = comparison_df.pivot_table(index='aspect', columns='sentiment', values='count', fill_value=0)
            
            categories = ['Positive', 'Negative', 'Neutral']
            for sent in categories:
                if sent not in radar_df.columns:
                    radar_df[sent] = 0
            radar_df = radar_df[categories]

            fig = go.Figure()

            for aspect in radar_df.index:
                # --- NEW: Create custom hover text for each point ---
                hover_text = [f"{count} {cat} mentions" for cat, count in zip(categories, radar_df.loc[aspect].values)]

                fig.add_trace(go.Scatterpolar(
                    r=radar_df.loc[aspect].values,
                    theta=categories,
                    fill='toself',
                    name=aspect,
                    hoverinfo='name+text', # Show aspect name and our custom text
                    text=hover_text        # Assign the custom text
                ))
            
            # Replace zero values with a very small number for log scale compatibility
            radar_df.replace(0, 0.1, inplace=True)

            # --- UPDATED: Hide axis labels for a cleaner look ---
            fig.update_layout(
              polar=dict(
                radialaxis=dict(
                  visible=True,
                  type='log',
                  showticklabels=False, # Hide the numbers
                  showline=False       # Hide the axis lines
                )),
              showlegend=True,
              title="Sentiment Profile Comparison (Log Scale)"
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Please select at least two aspects to generate a comparison.")
    else:
        st.warning("No aspects detected to compare.")

if __name__ == "__main__":
    main()
