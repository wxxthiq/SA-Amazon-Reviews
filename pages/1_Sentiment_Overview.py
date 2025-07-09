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
from utils.database_utils import (
    connect_to_db,
    get_product_details,
    get_reviews_for_product,
    get_product_date_range,
    get_single_review_details
)

# --- Page Configuration and State Initialization ---
st.set_page_config(layout="wide", page_title="Sentiment Overview")

@st.cache_resource
def load_spacy_model():
    return spacy.load("en_core_web_sm")

nlp = load_spacy_model()

if 'selected_review_id' not in st.session_state:
    st.session_state.selected_review_id = None

# --- Main App Logic ---
def main():
    st.title("📊 Sentiment Overview")
    
    DB_PATH = "amazon_reviews_top100.duckdb"
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
        st.image(image_urls[0] if image_urls else "https://via.placeholder.com/200", use_container_width=True)
        if image_urls:
            with st.popover("🖼️ View Image Gallery"):
                st.image(image_urls, use_container_width=True)
    with right_col:
        st.header(product_details['product_title'])
        st.caption(f"Category: {product_details['category']} | Store: {product_details['store']}")
        m_col1, m_col2 = st.columns(2)
        m_col1.metric("Average Rating", f"{product_details.get('average_rating', 0):.2f} ⭐")
        m_col2.metric("Filtered Reviews", f"{len(chart_data):,}")
        st.markdown("---")
        dist_col1, dist_col2 = st.columns(2)
        with dist_col1:
            st.markdown("**Rating Distribution**")
            if not chart_data.empty:
                rating_counts = chart_data['rating'].value_counts().reindex(range(1, 6), fill_value=0)
                total_ratings = len(chart_data)
                for rating in range(5, 0, -1):
                    count = rating_counts.get(rating, 0)
                    percentage = (count / total_ratings * 100) if total_ratings > 0 else 0
                    st.text(f"{rating} ⭐: {percentage:.1f}% ({count})")
                    st.progress(int(percentage))
        with dist_col2:
            st.markdown("**Sentiment Distribution**")
            if not chart_data.empty:
                sentiment_counts = chart_data['sentiment'].value_counts()
                total_sentiments = len(chart_data)
                sentiment_colors = {"Positive": "green", "Neutral": "grey", "Negative": "red"}
                for sentiment in ['Positive', 'Neutral', 'Negative']:
                    count = sentiment_counts.get(sentiment, 0)
                    percentage = (count / total_sentiments * 100) if total_sentiments > 0 else 0
                    st.markdown(f":{sentiment_colors.get(sentiment, 'default')}[{sentiment}]: {percentage:.1f}% ({count})")
                    st.progress(int(percentage))
    if chart_data.empty:
        st.warning("No reviews match the selected filters.")
        st.stop()
    st.info(f"Displaying analysis for **{len(chart_data)}** reviews matching your criteria.")

    # --- ASPECT SENTIMENT SUMMARY (WITH ENHANCED EXTRACTION) ---
    st.markdown("---")
    st.markdown("### 🔎 Aspect Sentiment Summary")
    st.caption("A summary of sentiment towards the most common product features (aspects).")

    @st.cache_data
    def get_aspect_summary_with_chunks(data):
        all_aspects = []
        # ** KEY CHANGE: New function to clean noun chunks **
        def clean_chunk(chunk):
            cleaned_tokens = []
            for token in chunk:
                # Keep nouns, proper nouns, and adjectives; remove determiners and pronouns
                if token.pos_ in ['NOUN', 'PROPN', 'ADJ']:
                    cleaned_tokens.append(token.lemma_.lower())
            return " ".join(cleaned_tokens)

        for doc in nlp.pipe(data['text'].astype(str)):
            for chunk in doc.noun_chunks:
                cleaned = clean_chunk(chunk)
                if cleaned and len(cleaned) > 2: # Ensure the cleaned chunk is not empty
                    all_aspects.append(cleaned)
        
        if not all_aspects:
            return pd.DataFrame()
            
        top_aspects = [aspect for aspect, freq in Counter(all_aspects).most_common(7)]

        aspect_sentiments = []
        for aspect in top_aspects:
            # Find reviews that contain any word from the aspect phrase
            search_regex = r'\b(' + '|'.join(re.escape(word) for word in aspect.split()) + r')\b'
            aspect_reviews = data[data['text'].str.contains(search_regex, case=False, na=False)]
            
            for text in aspect_reviews['text']:
                window = str(text).lower()[max(0, str(text).lower().find(aspect)-50):min(len(text), str(text).lower().find(aspect)+len(aspect)+50)]
                polarity = TextBlob(window).sentiment.polarity
                sentiment_cat = 'Positive' if polarity > 0.1 else 'Negative' if polarity < -0.1 else 'Neutral'
                aspect_sentiments.append({'aspect': aspect, 'sentiment': sentiment_cat})
        
        if not aspect_sentiments:
            return pd.DataFrame()
            
        return pd.DataFrame(aspect_sentiments)

    aspect_summary_df = get_aspect_summary_with_chunks(chart_data)

    if not aspect_summary_df.empty:
        # (The chart rendering code is unchanged)
        # ...
        summary_chart_data = aspect_summary_df.groupby(['aspect', 'sentiment']).size().reset_index(name='count')
        chart = alt.Chart(summary_chart_data).mark_bar().encode(
            x=alt.X('count:Q', title='Number of Mentions'),
            y=alt.Y('aspect:N', sort='-x', title='Aspect'),
            color=alt.Color('sentiment:N', scale=alt.Scale(domain=['Positive', 'Neutral', 'Negative'], range=['#1a9850', '#cccccc', '#d73027']), legend=alt.Legend(title="Sentiment")),
            yOffset='sentiment:N'
        ).configure_axis(grid=False).configure_view(strokeWidth=0)
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("Not enough data to generate an aspect summary for the current filters.")

    if st.button("Perform Detailed Aspect Analysis 🔎"):
        st.switch_page("pages/4_Aspect_Analysis.py")
    
    # --- KEYWORD ANALYSIS SECTION (WITH N-GRAMS) ---
    st.markdown("---")
    st.markdown("### ☁️ Keyword & Phrase Summary")
    st.caption("The most common terms found in positive and negative reviews. Use the options below to explore single words or multi-word phrases.")

    col1, col2 = st.columns([1,1])
    with col1:
        max_words = st.slider("Max Terms in Cloud:", min_value=5, max_value=50, value=15)
    with col2:
        ngram_level = st.radio("Term Type:", ("Single Words", "Bigrams", "Trigrams"), index=0, horizontal=True)

    # Helper function to generate n-grams
    def get_top_ngrams(corpus, n=None, ngram_range=(1,1)):
        vec = CountVectorizer(ngram_range=ngram_range, stop_words='english').fit(corpus)
        bag_of_words = vec.transform(corpus)
        sum_words = bag_of_words.sum(axis=0) 
        words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
        words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
        return words_freq[:n]

    ngram_range = {
        "Single Words": (1,1),
        "Bigrams": (2,2),
        "Trigrams": (3,3)
    }.get(ngram_level, (1,1))

    wc_col1, wc_col2 = st.columns(2)

    with wc_col1:
        st.markdown("#### Positive Reviews")
        pos_text = chart_data[chart_data["sentiment"]=="Positive"]["text"].dropna()
        if not pos_text.empty:
            top_pos_grams = get_top_ngrams(pos_text, n=max_words, ngram_range=ngram_range)
            pos_freq_dict = dict(top_pos_grams)
            if pos_freq_dict:
                wordcloud_pos = WordCloud(
                    stopwords=STOPWORDS, background_color="white", width=800, height=400, colormap='Greens'
                ).generate_from_frequencies(pos_freq_dict)
                fig, ax = plt.subplots()
                ax.imshow(wordcloud_pos, interpolation='bilinear')
                ax.axis("off")
                st.pyplot(fig)
            else:
                st.info("No terms found for this n-gram level.")
        else:
            st.info("No positive reviews to analyze.")

    with wc_col2:
        st.markdown("#### Negative Reviews")
        neg_text = chart_data[chart_data["sentiment"]=="Negative"]["text"].dropna()
        if not neg_text.empty:
            top_neg_grams = get_top_ngrams(neg_text, n=max_words, ngram_range=ngram_range)
            neg_freq_dict = dict(top_neg_grams)
            if neg_freq_dict:
                wordcloud_neg = WordCloud(
                    stopwords=STOPWORDS, background_color="white", width=800, height=400, colormap='Reds'
                ).generate_from_frequencies(neg_freq_dict)
                fig, ax = plt.subplots()
                ax.imshow(wordcloud_neg, interpolation='bilinear')
                ax.axis("off")
                st.pyplot(fig)
            else:
                st.info("No terms found for this n-gram level.")
        else:
            st.info("No negative reviews to analyze.")
    
    # --- Navigation Buttons ---
    st.markdown("---")
    if st.button("Perform Detailed Keyword Analysis 🔑"):
        st.switch_page("pages/3_Keyword_Analysis.py")
    
    st.markdown("### Rating vs. Text Discrepancy")
    plot_col, review_col = st.columns([2, 1])
    with plot_col:
        # ... (code omitted for brevity)
        chart_data['discrepancy'] = (chart_data['text_polarity'] - ((chart_data['rating'] - 3) / 2.0)).abs()
        fig = px.scatter(chart_data, x="rating_jittered", y="text_polarity_jittered", color="discrepancy", color_continuous_scale=px.colors.sequential.Viridis, hover_name='review_title')
        fig.update_layout(clickmode='event+select')
        fig.update_traces(marker_size=10)
        selected_points = plotly_events(fig, click_event=True, key="plotly_event_selector")
        if selected_points and 'pointIndex' in selected_points[0]:
            point_index = selected_points[0]['pointIndex']
            if point_index < len(chart_data):
                clicked_id = chart_data.iloc[point_index]['review_id']
                if st.session_state.selected_review_id != clicked_id:
                    st.session_state.selected_review_id = clicked_id
                    st.rerun()
    with review_col:
        # ... (code omitted for brevity)
        if st.session_state.selected_review_id:
            if st.session_state.selected_review_id in chart_data['review_id'].values:
                st.markdown("#### Selected Review Details")
                review_details = get_single_review_details(conn, st.session_state.selected_review_id)
                if review_details is not None:
                    st.subheader(review_details['review_title'])
                    st.caption(f"Reviewed on: {review_details['date']}")
                    st.markdown(f"> {review_details['text']}")
                if st.button("Close Review", key="close_review_button"):
                    st.session_state.selected_review_id = None
                    st.rerun()
    st.markdown("---")
    st.markdown("### Trends Over Time")
    time_granularity = st.radio("Select time period:", ("Monthly", "Weekly", "Daily"), index=0, horizontal=True, label_visibility="collapsed")
    # ... (code for time charts omitted for brevity)
    time_df = chart_data.copy()
    time_df['date'] = pd.to_datetime(time_df['date'])
    if time_granularity == 'Monthly':
        time_df['period'] = time_df['date'].dt.to_period('M').dt.start_time
    elif time_granularity == 'Weekly':
        time_df['period'] = time_df['date'].dt.to_period('W').dt.start_time
    else: # Daily
        time_df['period'] = time_df['date'].dt.date
    t_col1, t_col2 = st.columns(2)
    with t_col1:
        st.markdown("#### Rating Distribution Over Time")
        rating_counts_over_time = time_df.groupby(['period', 'rating']).size().reset_index(name='count')
        if not rating_counts_over_time.empty:
            rating_stream_chart = px.area(rating_counts_over_time, x='period', y='count', color='rating', title=f"Volume of Reviews by Star Rating", color_discrete_map={5: '#1a9850', 4: '#91cf60', 3: '#d9ef8b', 2: '#fee08b', 1: '#d73027'}, category_orders={"rating": [5, 4, 3, 2, 1]})
            st.plotly_chart(rating_stream_chart, use_container_width=True)
            
    with t_col2:
        st.markdown("#### Sentiment Volume Over Time")
        sentiment_counts_over_time = time_df.groupby(['period', 'sentiment']).size().reset_index(name='count')
        
        if not sentiment_counts_over_time.empty:
            # --- NEW: Calculate Moving Average ---
            # Sort values to ensure correct rolling calculation
            sentiment_counts_over_time = sentiment_counts_over_time.sort_values('period')
            # Calculate a 7-period moving average for each sentiment category
            sentiment_counts_over_time['moving_average'] = sentiment_counts_over_time.groupby('sentiment')['count'].transform(lambda x: x.rolling(7, min_periods=1).mean())

            # --- UPDATED: Build chart with Plotly Graph Objects ---
            fig = go.Figure()

            sentiments = ['Positive', 'Neutral', 'Negative']
            colors = {'Positive': '#1a9850', 'Neutral': '#cccccc', 'Negative': '#d73027'}
            
            for sentiment in sentiments:
                df_sentiment = sentiment_counts_over_time[sentiment_counts_over_time['sentiment'] == sentiment]
                if not df_sentiment.empty:
                    # Original volume area chart (semi-transparent)
                    fig.add_trace(go.Scatter(
                        x=df_sentiment['period'],
                        y=df_sentiment['count'],
                        mode='lines',
                        fill='tozeroy',
                        name=f"{sentiment} Volume",
                        line=dict(width=0.5, color=colors[sentiment]),
                        opacity=0.4
                    ))
                    # Moving average trendline
                    fig.add_trace(go.Scatter(
                        x=df_sentiment['period'],
                        y=df_sentiment['moving_average'],
                        mode='lines',
                        name=f"{sentiment} Trend",
                        line=dict(width=2, color=colors[sentiment])
                    ))
            
            fig.update_layout(
                title=f"Sentiment Breakdown Per {time_granularity.replace('ly', '')}",
                yaxis_title="Number of Reviews"
            )
            st.plotly_chart(fig, use_container_width=True)

    # --- Navigation to Review Explorer ---
    st.markdown("---")
    st.subheader("📝 Browse Individual Reviews")
    if st.button("Explore All Reviews"):
        st.switch_page("pages/2_Review_Explorer.py")

if __name__ == "__main__":
    main()
