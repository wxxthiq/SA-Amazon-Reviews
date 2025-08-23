# pages/1_Sentiment_Overview.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import altair as alt
from datetime import datetime
from streamlit_plotly_events import plotly_events
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
import spacy
from collections import Counter
from textblob import TextBlob
import re
import plotly.graph_objects as go 
import json
from utils.database_utils import (
    connect_to_db,
    get_product_details,
    get_reviews_for_product,
    get_product_date_range,
    get_single_review_details,
    get_aspects_for_product # --- IMPORT THE NEW FUNCTION ---
)

st.set_page_config(layout="wide", page_title="Sentiment Overview")

def render_help_popover(title, what, how, learn):
    """Creates a standardized help popover next to a title."""
    with st.container():
        c1, c2 = st.columns([0.9, 0.1])
        with c1:
            st.markdown(f"**{title}**")
        with c2:
            with st.popover("ⓘ"):
                st.markdown("##### What am I looking at?")
                st.markdown(what)
                st.markdown("##### How do I use it?")
                st.markdown(how)
                st.markdown("##### What can I learn?")
                st.markdown(learn)
                
# --- NEW: Clickable Icon and State Management Function ---
def clickable_help_icon(topic: str):
    """Creates a clickable help icon that toggles a help topic in session state."""
    # Use a unique key for each button
    if st.button("ⓘ", key=f"help_{topic}", help=f"Click to see details about the {topic} chart"):
        # If the button is clicked, toggle the state for this topic
        if st.session_state.get('active_help_topic') == topic:
            st.session_state['active_help_topic'] = None # Hide if it's already active
        else:
            st.session_state['active_help_topic'] = topic # Show this topic
        
@st.cache_resource
def load_spacy_model():
    return spacy.load("en_core_web_sm")

nlp = load_spacy_model()
    
if 'selected_review_id' not in st.session_state:
    st.session_state.selected_review_id = None

# --- Main App Logic ---
def main():
    st.title("📊 Sentiment Overview")

    # ADD THIS CSS BLOCK
    st.markdown("""
        <style>
        .product-image-container {
            height: 350px; /* Adjust height as needed for this page */
            display: flex;
            align-items: center;
            justify-content: center;
            margin-bottom: 1rem;
        }
        .product-image-container img {
            max-height: 100%;
            max-width: 100%;
            object-fit: contain;
        }
        </style>
    """, unsafe_allow_html=True)
    DB_PATH = "amazon_reviews_final.duckdb"
    conn = connect_to_db(DB_PATH)
    
    if 'selected_product' not in st.session_state or st.session_state.selected_product is None:
        st.warning("Please select a product from the main search page first.")
        st.stop()
        
    selected_asin = st.session_state.selected_product
    product_details_df = get_product_details(conn, selected_asin)
    if product_details_df.empty:
        st.error("Could not find details for the selected product.")
        st.stop()
        
    product_details = product_details_df.iloc[0]
    
    # --- Sidebar Filters (WITH VERIFIED PURCHASE) ---
    st.sidebar.header("📊 Interactive Filters")
    min_date_db, max_date_db = get_product_date_range(conn, selected_asin)
    
    default_date_range = (min_date_db, max_date_db)
    default_ratings = [1, 2, 3, 4, 5]
    default_sentiments = ['Positive', 'Negative', 'Neutral']
    default_verified = "All"
    
    def reset_all_filters():
        st.session_state.date_filter = default_date_range
        st.session_state.rating_filter = default_ratings
        st.session_state.sentiment_filter = default_sentiments
        st.session_state.verified_filter = default_verified # Reset new filter
        st.session_state.selected_review_id = None
        
    def reset_selection():
        st.session_state.selected_review_id = None
    
    selected_date_range = st.sidebar.date_input("Filter by Date Range", value=default_date_range, key='date_filter', on_change=reset_selection)
    selected_ratings = st.sidebar.multiselect("Filter by Star Rating", options=default_ratings, default=default_ratings, key='rating_filter', on_change=reset_selection)
    selected_sentiments = st.sidebar.multiselect("Filter by Sentiment", options=default_sentiments, default=default_sentiments, key='sentiment_filter', on_change=reset_selection)
    selected_verified = st.sidebar.radio("Filter by Purchase Status", ["All", "Verified Only", "Not Verified"], index=0, key='verified_filter', on_change=reset_selection)
    
    st.sidebar.button("Reset All Filters", on_click=reset_all_filters, use_container_width=True)
    chart_data = get_reviews_for_product(conn, selected_asin, selected_date_range, tuple(selected_ratings), tuple(selected_sentiments), selected_verified)

    if st.button("⬅️ Back to Search"):
        st.session_state.selected_product = None
        st.session_state.selected_review_id = None
        st.switch_page("app.py")
    left_col, right_col = st.columns([1, 2])
    with left_col:
        image_urls_str = product_details.get('image_urls')
        image_urls = image_urls_str.split(',') if pd.notna(image_urls_str) and image_urls_str else []
        # --- MODIFIED SECTION ---
        thumbnail_url = image_urls[0] if image_urls else "https://via.placeholder.com/200"
        st.markdown(f"""
            <div class="product-image-container">
                <img src="{thumbnail_url}">
            </div>
        """, unsafe_allow_html=True)
        # --- END MODIFICATION ---
        if image_urls:
            with st.popover("🖼️ View Image Gallery"):
                st.image(image_urls, use_container_width=True)
    
        if pd.notna(product_details.get('description')):
            with st.expander("Description"):
                st.write(product_details['description'])
    
        # --- Navigation to Review Explorer ---
        if st.button("📝 Explore All Reviews"):
            st.switch_page("pages/2_Review_Explorer.py")
        if st.button("⚖️ Compare this Product"):
            st.switch_page("pages/5_Product_Comparison.py")
            
    with right_col:
        st.header(product_details['product_title'])
        #st.caption(f"Category: {product_details['category']} | Store: {product_details['store']}")
        # In main() -> with right_col:
        # --- NEW: 2x2 Grid for Metrics ---
        row1_col1, row1_col2 = st.columns(2)
        row2_col1, row2_col2 = st.columns(2)

        # --- Row 1, Col 1: Average Rating (Existing) ---
        with row1_col1:
            st.metric("Average Rating", f"{product_details.get('average_rating', 0):.2f} ⭐", help="The average star rating for this product across all reviews.")

        # --- Row 1, Col 2: Average Sentiment Score (New) ---
        with row1_col2:
            if not chart_data.empty:
                avg_sentiment_score = chart_data['sentiment_score'].mean()
                # Use an emoji to represent the score
                emoji = "😊" if avg_sentiment_score > 0.3 else "😐" if avg_sentiment_score > -0.3 else "😞"
                st.metric(
                    "Average Sentiment",
                    f"{avg_sentiment_score:.2f} {emoji}",
                    help="The average sentiment score of the review text, from -1.0 (very negative) to 1.0 (very positive)."
                )
            else:
                st.metric("Average Sentiment", "N/A")

        # --- Row 2, Col 1: Reviewer Consensus (Existing) ---
        with row2_col1:
             if not chart_data.empty and len(chart_data) > 1:
                rating_std_dev = chart_data['rating'].std()
                 
                def get_rating_consensus(std_dev):
                    if std_dev < 1.1:
                        return "✅ Consistent"  # Low deviation = high agreement
                    elif std_dev < 1.4:
                        return "↔️ Mixed"      # Medium deviation = some disagreement
                    else:
                        return "⚠️ Polarizing" # High deviation = disagreement
                        
                consensus_text = get_rating_consensus(rating_std_dev)
                st.metric(
                    "Reviewer Consensus",
                    consensus_text,
                    help="Measures agreement in star ratings. 'Consistent' means ratings are similar; 'Polarizing' means there are many high and low ratings."
                )
             else:
                 st.metric("Reviewer Consensus", "N/A")


        # --- Row 2, Col 2: Verified Purchases (New) ---
        with row2_col2:
            if not chart_data.empty:
                verified_percentage = (chart_data['verified_purchase'].sum() / len(chart_data)) * 100
                st.metric(
                    "Verified Purchases",
                    f"{verified_percentage:.1f}%",
                    help="The percentage of reviews left by customers with a confirmed purchase of this product."
                )
            else:
                st.metric("Verified Purchases", "N/A")

        st.info(f"**{len(chart_data):,}** reviews match your current filters.")
        st.markdown("---")
        # --- MODIFIED: Back to a 2-column layout ---
        dist_col1, dist_col2 = st.columns(2)
        # --- Column 1: Rating Distribution ---
        with dist_col1:
            # --- Title and Icon ---
            title_c1, icon_c1 = st.columns([0.9, 0.1])
            with title_c1:
                st.markdown("**⭐ Rating Distribution**")
            with icon_c1:
                clickable_help_icon("Rating") # The button to toggle the help text
                
            if st.session_state.get('active_help_topic') == "Rating":
                with st.container(border=True):
                    st.markdown("##### What am I looking at?")
                    st.markdown("This chart shows the breakdown of reviews by the star rating (1 to 5 stars) that reviewers gave.")
                    st.markdown("##### How do I use it?")
                    st.markdown("Hover over a segment to see the specific rating, the number of reviews, and its percentage of the total.")
                    st.markdown("##### What can I learn?")
                    st.markdown("Quickly see if a product is generally well-liked (large green segments) or has significant issues (visible red segments).")

            # --- Chart Display ---
            if not chart_data.empty:
                rating_counts_df = chart_data['rating'].value_counts().reset_index()
                rating_counts_df.columns = ['rating', 'count']
                rating_counts_df['rating_cat'] = rating_counts_df['rating'].astype(str) + ' ⭐'
                
                # --- NEW: Calculate percentage for tooltip ---
                total_reviews = rating_counts_df['count'].sum()
                rating_counts_df['percentage'] = rating_counts_df['count'] / total_reviews if total_reviews > 0 else 0
            
                bar_chart = alt.Chart(rating_counts_df).mark_bar().encode(
                    x=alt.X('sum(count)', stack='normalize', axis=alt.Axis(title='Percentage', format='%')),
                    color=alt.Color('rating_cat:N',
                                    scale=alt.Scale(domain=['5 ⭐', '4 ⭐', '3 ⭐', '2 ⭐', '1 ⭐'],
                                                    range=['#2ca02c', '#98df8a', '#ffdd71', '#ff9896', '#d62728']),
                                    legend=alt.Legend(title="Rating")),
                    order=alt.Order('rating_cat', sort='descending'),
                    # --- MODIFIED: Added percentage to tooltip ---
                    tooltip=[
                        alt.Tooltip('rating_cat', title='Rating'),
                        alt.Tooltip('count', title='Reviews'),
                        alt.Tooltip('percentage:Q', title='Share', format='.1%')
                    ]
                ).properties(height=150)
                st.altair_chart(bar_chart, use_container_width=True)

        # --- Column 2: Sentiment Distribution ---
        with dist_col2:
            # --- Title and Icon ---
            title_c2, icon_c2 = st.columns([0.9, 0.1])
            with title_c2:
                st.markdown("**😊 Sentiment Distribution**")
            with icon_c2:
                clickable_help_icon("Sentiment") # The button to toggle the help text

            # --- Conditional Help Text Container ---
            if st.session_state.get('active_help_topic') == "Sentiment":
                with st.container(border=True):
                    st.markdown("##### What am I looking at?")
                    st.markdown("This chart shows the breakdown of reviews by their automatically detected sentiment (Positive, Negative, or Neutral).")
                    st.markdown("##### How do I use it?")
                    st.markdown("Hover over a segment to see the sentiment, the number of reviews, and its percentage of the total.")
                    st.markdown("##### What can I learn?")
                    st.markdown("Understand the overall feeling of the reviews. A large red segment might indicate widespread problems.")

            # --- Chart Display ---
            if not chart_data.empty:
                sentiment_counts_df = chart_data['sentiment'].value_counts().reset_index()
                sentiment_counts_df.columns = ['sentiment', 'count']
                
                # --- NEW: Calculate percentage for tooltip ---
                total_sentiments = sentiment_counts_df['count'].sum()
                sentiment_counts_df['percentage'] = sentiment_counts_df['count'] / total_sentiments if total_sentiments > 0 else 0
            
                bar_chart = alt.Chart(sentiment_counts_df).mark_bar().encode(
                    x=alt.X('sum(count)', stack='normalize', axis=alt.Axis(title='Percentage', format='%')),
                    color=alt.Color('sentiment:N',
                                    scale=alt.Scale(domain=['Positive', 'Neutral', 'Negative'],
                                                    range=['#1a9850', '#cccccc', '#d73027']),
                                    legend=alt.Legend(title="Sentiment")),
                    order=alt.Order('sentiment', sort='descending'),
                    # --- MODIFIED: Added percentage to tooltip ---
                    tooltip=[
                        alt.Tooltip('sentiment', title='Sentiment'),
                        alt.Tooltip('count', title='Reviews'),
                        alt.Tooltip('percentage:Q', title='Share', format='.1%')
                    ]
                ).properties(height=150)
                st.altair_chart(bar_chart, use_container_width=True)
                
    if chart_data.empty:
        st.warning("No reviews match the selected filters.")
        st.stop()
        
    st.markdown("---")
    col1, col2 = st.columns([3, 2])
    with col1:
        # --- MODIFIED: Title with an integrated help popover ---
        title_c, popover_c = st.columns([0.9, 0.1])
        with title_c:
            st.markdown("### ☁️ Keyword & Phrase Summary")
        with popover_c:
            with st.popover("ⓘ"):
                st.markdown("##### What am I looking at?")
                st.markdown("These word clouds show the most frequent words or phrases found in positive and negative reviews. The larger the word, the more often it was mentioned.")
                st.markdown("##### How do I use it?")
                st.markdown("Use the 'Advanced Settings' expander to switch between single words, two-word phrases (bigrams), or three-word phrases (trigrams) to find more specific insights.")
                st.markdown("##### What can I learn?")
                st.markdown("Quickly identify the key terms customers use to praise or complain about the product. This helps you spot common themes at a glance.")
        with st.expander("Advanced Settings"):
            control_col1, control_col2 = st.columns(2)
            with control_col1:
                max_words = st.slider("Max Terms in Cloud:", min_value=5, max_value=50, value=15, key="keyword_slider")
            with control_col2:
                ngram_level = st.radio("Term Type:", ("Single Words", "2 Words", "3 Words"), index=0, horizontal=True, key="ngram_radio")

        def get_top_ngrams(corpus, n=None, ngram_range=(1,1)):
            vec = CountVectorizer(ngram_range=ngram_range, stop_words='english').fit(corpus)
            bag_of_words = vec.transform(corpus)
            sum_words = bag_of_words.sum(axis=0) 
            words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
            words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
            return words_freq[:n]

        ngram_range = {"Single Words": (1,1), "Bigrams": (2,2), "Trigrams": (3,3)}.get(ngram_level, (1,1))
        wc_col1, wc_col2 = st.columns(2)
        with wc_col1:
            st.markdown("#### Positive Reviews")
            pos_text = chart_data[chart_data["sentiment"]=="Positive"]["text"].dropna()
            if not pos_text.empty:
                top_pos_grams = get_top_ngrams(pos_text, n=max_words, ngram_range=ngram_range)
                if top_pos_grams:
                    pos_freq_dict = dict(top_pos_grams)
                    wordcloud_pos = WordCloud(stopwords=STOPWORDS, background_color="white", colormap='Greens').generate_from_frequencies(pos_freq_dict)
                    fig, ax = plt.subplots()
                    ax.imshow(wordcloud_pos, interpolation='bilinear')
                    ax.axis("off")
                    st.pyplot(fig)
                else:
                    st.caption("No terms found.")
            else:
                st.caption("No positive reviews.")
        
        with wc_col2:
            st.markdown("#### Negative Reviews")
            neg_text = chart_data[chart_data["sentiment"]=="Negative"]["text"].dropna()
            if not neg_text.empty:
                top_neg_grams = get_top_ngrams(neg_text, n=max_words, ngram_range=ngram_range)
                if top_neg_grams:
                    neg_freq_dict = dict(top_neg_grams)
                    wordcloud_neg = WordCloud(stopwords=STOPWORDS, background_color="white", colormap='Reds').generate_from_frequencies(neg_freq_dict)
                    fig, ax = plt.subplots()
                    ax.imshow(wordcloud_neg, interpolation='bilinear')
                    ax.axis("off")
                    st.pyplot(fig)
                else:
                    st.caption("No terms found.")
            else:
                st.caption("No negative reviews.")

    # --- Column 2: Aspect Sentiment Analysis ---
    with col2:
        # --- MODIFIED: Title and Icon ---
        title_c, icon_c = st.columns([0.9, 0.1])
        with title_c:
            st.markdown("### 🔎 Key Aspect Summary")
        with icon_c:
            # This button will toggle the help text below
            clickable_help_icon("Aspect")

        # --- MODIFIED: Conditional Help Text Container ---
        if st.session_state.get('active_help_topic') == "Aspect":
            with st.container(border=True):
                st.markdown("##### What am I looking at?")
                st.markdown("This chart identifies key product features (aspects) and shows the sentiment breakdown for each.")
                st.markdown("##### How do I use it?")
                st.markdown("Use the dropdown to sort aspects by popularity, or by which are most positive or negative. The sliders let you change how many aspects are displayed and filter out rarely-mentioned terms.")
                st.markdown("##### What can I learn?")
                st.markdown("Pinpoint specific product strengths and weaknesses. An aspect with a large red bar is a clear area for concern.")

        # --- Original Controls and Chart Logic (no changes needed below this line) ---

        with st.expander("Advanced Settings"):
            sort_option = st.selectbox(
                "Sort aspects by:",
                ("Most Discussed", "Most Positive", "Most Negative", "Most Controversial"),
                key="aspect_sort_selector"
            )
            num_aspects_to_show = st.slider(
                "Select number of top aspects to display:",
                min_value=3, max_value=15, value=5, key="overview_aspect_slider"
            )
            smart_threshold = max(3, min(10, int(len(chart_data) * 0.01)))
            min_mentions = st.slider(
                "Aspect Mention Threshold",
                min_value=1,
                max_value=50,
                value=smart_threshold,
                help="Filters out aspects mentioned fewer than this many times to reduce noise."
            )
        aspect_df = get_aspects_for_product(
        conn, selected_asin, selected_date_range,
        tuple(selected_ratings), tuple(selected_sentiments), selected_verified
        )
        if not aspect_df.empty:
            # --- UPDATED: Use the dynamic min_mentions from the slider ---
            aspect_counts = aspect_df['aspect'].value_counts()
            significant_aspects = aspect_counts[aspect_counts >= min_mentions].index.tolist()

            if not significant_aspects:
                st.warning(f"No aspects were mentioned at least {min_mentions} times. Try lowering the threshold slider.")
                st.stop()

            filtered_aspect_df = aspect_df[aspect_df['aspect'].isin(significant_aspects)]

            # Data Processing and Sorting Logic
            sentiment_counts = filtered_aspect_df.groupby(['aspect', 'sentiment']).size().reset_index(name='count')
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

            # Ensure we don't try to show more aspects than are available after filtering
            num_aspects_to_show = min(num_aspects_to_show, len(pivot_df))
            top_aspects_sorted = pivot_df.nlargest(num_aspects_to_show, sort_field).index.tolist()
            
            if not top_aspects_sorted:
                 st.warning("No aspects to display with the current settings.")
                 st.stop()

            top_aspects_df = sentiment_counts[sentiment_counts['aspect'].isin(top_aspects_sorted)]

            # The Charting Logic
            chart = alt.Chart(top_aspects_df).mark_bar().encode(
                y=alt.Y('aspect:N', title=None, sort=alt.EncodingSortField(field=sort_field, op="sum", order=sort_order)),
                x=alt.X('sum(count):Q', stack="normalize", title="Sentiment Distribution", axis=alt.Axis(format='%')),
                color=alt.Color('sentiment:N', scale=alt.Scale(domain=['Positive', 'Neutral', 'Negative'], range=['#1a9850', '#cccccc', '#d73027']), legend=alt.Legend(title="Sentiment")),
                tooltip=[alt.Tooltip('aspect', title='Aspect'), alt.Tooltip('sentiment', title='Sentiment'), alt.Tooltip('sum(count):Q', title='Review Count')]
            ).properties(title=f"Top {num_aspects_to_show} Aspects (Sorted by {sort_option})", height = 400)
            st.altair_chart(chart, use_container_width=True)
            
        else:
            st.info("No aspect data found for the selected filters.")
            
    # --- Navigation Buttons ---
    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        if st.button("Perform Detailed Keyword Analysis 🔑", use_container_width=True):
            st.switch_page("pages/3_Keyword_Analysis.py")
    with btn_col2:
        if st.button("Perform Detailed Aspect Analysis 🔎", use_container_width=True):
            st.switch_page("pages/4_Aspect_Analysis.py")

    # --- TRENDS OVER TIME ---
    st.markdown("---")
    # --- MODIFIED: Title with an integrated help popover ---
    title_c, popover_c = st.columns([0.9, 0.1])
    with title_c:
        st.markdown("### 🗓️ Trends Over Time")
    with popover_c:
        with st.popover("ⓘ"):
            st.markdown("##### What am I looking at?")
            st.markdown("These charts analyze how the volume of review ratings and sentiment has evolved over time.")
            st.markdown("##### How do I use it?")
            st.markdown("Use the 'Select time period' radio buttons to change the granularity (e.g., from monthly to weekly). Use the toggles to overlay an average trendline. The 'Line Chart' view is best for comparing individual trends, while the 'Area Chart' view shows the overall distribution.")
            st.markdown("##### What can I learn?")
            st.markdown("Spot seasonal patterns, the impact of a product change, or whether sentiment is generally improving or declining over time.")
            
    time_granularity = st.radio(
        "Select time period:",
        ("Monthly", "Weekly", "Daily"),
        index=0, horizontal=True
    )
    
    # Prepare the data once for all charts
    time_df = chart_data.copy()
    time_df['date'] = pd.to_datetime(time_df['date'])
    if time_granularity == 'Monthly':
        time_df['period'] = time_df['date'].dt.to_period('M').dt.start_time
    elif time_granularity == 'Weekly':
        time_df['period'] = time_df['date'].dt.to_period('W').dt.start_time
    else: # Daily
        time_df['period'] = time_df['date'].dt.date
    
    rating_counts_over_time = time_df.groupby(['period', 'rating']).size().reset_index(name='count')
    sentiment_counts_over_time = time_df.groupby(['period', 'sentiment']).size().reset_index(name='count')
    tab1, tab2 = st.tabs(["📈 Line Chart View", "📊 Area Chart View"])
    # --- Tab 1: Line Chart View ---
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            show_rating_trend = st.toggle('Show Average Rating Trend', key='line_rating_trend')
            if not rating_counts_over_time.empty:
                
                # --- FIX: Use category_orders for sorting and update the color map ---
                fig = px.line(
                    rating_counts_over_time, x='period', y='count', color='rating',
                    labels={'period': 'Date', 'count': 'Number of Reviews', 'rating': 'Star Rating'},
                    color_discrete_map={
                        5: '#1a9850', 4: '#91cf60', 3: 'yellow', 
                        2: 'orange', # <-- Set Rating 2 to orange
                        1: '#d73027'
                    },
                    # Enforce descending sort order for the legend
                    category_orders={"rating": [5, 4, 3, 2, 1]}
                )
    
                if show_rating_trend:
                    avg_rating_trend = time_df.groupby('period')['rating'].mean().reset_index()
                    fig.add_trace(go.Scatter(
                        x=avg_rating_trend['period'], y=avg_rating_trend['rating'], 
                        mode='lines', name='Average Rating', yaxis='y2',
                        line=dict(dash='dash', color='blue')
                    ))
                    fig.update_layout(
                        title_text="Trend and Average Rating",
                        yaxis2=dict(title='Avg Rating', overlaying='y', side='right', range=[1, 5]),
                        legend=dict(
                            orientation="h", # Horizontal legend
                            yanchor="bottom",
                            y=1.02,          # Positioned above the chart
                            xanchor="right",
                            x=1))
                else:
                    fig.update_layout(title_text="Volume by Rating")
                
                st.plotly_chart(fig, use_container_width=True)
        with col2:
            show_sentiment_trend = st.toggle('Show Average Sentiment Trend', key='line_sentiment_trend')
            if not sentiment_counts_over_time.empty:
                if show_sentiment_trend:
                    avg_sentiment_trend = time_df.groupby('period')['sentiment_score'].mean().reset_index()
                    fig = px.line(
                        sentiment_counts_over_time, x='period', y='count', color='sentiment',
                        labels={'period': 'Date', 'count': 'Number of Reviews', 'sentiment': 'Sentiment'},
                        color_discrete_map={'Positive': '#1a9850', 'Neutral': '#cccccc', 'Negative': '#d73027'},
                        category_orders={"sentiment": ["Positive", "Neutral", "Negative"]}
                    )
                    fig.add_trace(go.Scatter(x=avg_sentiment_trend['period'], y=avg_sentiment_trend['sentiment_score'], mode='lines', name='Avg. Sentiment', yaxis='y2', line=dict(dash='dash')))
                    fig.update_layout(title_text="Trend and Average Sentiment", yaxis2=dict(title='Avg Sentiment', overlaying='y', side='right', range=[-1, 1]))
                else:
                    fig = px.line(
                        sentiment_counts_over_time, x='period', y='count', color='sentiment', title="Volume by Sentiment",
                        labels={'period': 'Date', 'count': 'Number of Reviews', 'sentiment': 'Sentiment'},
                        color_discrete_map={'Positive': '#1a9850', 'Neutral': '#cccccc', 'Negative': '#d73027'},
                        category_orders={"sentiment": ["Positive", "Neutral", "Negative"]}
                    )
                st.plotly_chart(fig, use_container_width=True)
    # --- Tab 2: Area Chart View ---
    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            if not rating_counts_over_time.empty:
                fig_area_rating = px.area(
                    rating_counts_over_time, x='period', y='count', color='rating', title="Distribution by Rating",
                    labels={'period': 'Date', 'count': 'Number of Reviews', 'rating': 'Star Rating'},
                    color_discrete_map={5: '#1a9850', 4: '#91cf60', 3: '#d9ef8b', 2: '#fee08b', 1: '#d73027'},
                    category_orders={"rating": [5, 4, 3, 2, 1]}
                )
                st.plotly_chart(fig_area_rating, use_container_width=True)
    
        with col2:
            if not sentiment_counts_over_time.empty:
                fig_area_sentiment = px.area(
                    sentiment_counts_over_time, x='period', y='count', color='sentiment', title="Distribution by Sentiment",
                    labels={'period': 'Date', 'count': 'Number of Reviews', 'sentiment': 'Sentiment'},
                    color_discrete_map={'Positive': '#1a9850', 'Neutral': '#cccccc', 'Negative': '#d73027'},
                    category_orders={"sentiment": ["Positive", "Neutral", "Negative"]}
                )
                st.plotly_chart(fig_area_sentiment, use_container_width=True)


if __name__ == "__main__":
    main()
