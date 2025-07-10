# pages/5_Product_Comparison.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from utils.database_utils import connect_to_db, get_product_details, get_reviews_for_product, get_product_date_range
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from datetime import datetime
import spacy
from textblob import TextBlob
import re

# --- Page Configuration and Model Loading ---
st.set_page_config(layout="wide", page_title="Advanced Product Comparison")
DB_PATH = "amazon_reviews_top100.duckdb"
conn = connect_to_db(DB_PATH)

@st.cache_resource
def load_spacy_model():
    """Loads the spaCy model for NLP tasks."""
    return spacy.load("en_core_web_sm")

nlp = load_spacy_model()

# --- Helper Functions ---

@st.cache_data
def get_review_data_for_asins(_conn, asins, date_range, rating_filter, sentiment_filter, verified_filter):
    """Fetches and caches review data for a list of ASINs based on universal filters."""
    review_data = {}
    for asin in asins:
        review_data[asin] = get_reviews_for_product(
            _conn, asin, date_range, tuple(rating_filter), tuple(sentiment_filter), verified_filter
        )
    return review_data

@st.cache_data
def get_comparative_aspect_sentiments(_review_data_cache, top_n_aspects):
    """
    Performs aspect-based sentiment analysis across all compared products.
    This function replicates the logic from the Aspect Analysis page.
    """
    all_aspect_sentiments = []
    
    # First, find the most common aspects across all products combined
    all_text_corpus = pd.concat([df['text'] for df in _review_data_cache.values()]).astype(str)
    
    all_aspects = []
    def clean_chunk(chunk):
        return " ".join(token.lemma_.lower() for token in chunk if token.pos_ in ['NOUN', 'PROPN', 'ADJ'])

    for doc in nlp.pipe(all_text_corpus):
        for chunk in doc.noun_chunks:
            cleaned = clean_chunk(chunk)
            if cleaned and len(cleaned) > 2:
                all_aspects.append(cleaned)
    
    if not all_aspects:
        return pd.DataFrame()
        
    top_overall_aspects = [aspect for aspect, freq in Counter(all_aspects).most_common(top_n_aspects)]

    # Now, calculate sentiment for these aspects within each product's reviews
    for asin, df in _review_data_cache.items():
        if df.empty:
            continue
        
        product_title = get_product_details(conn, asin).iloc[0]['product_title']
        
        for aspect in top_overall_aspects:
            for text in df['text']:
                if re.search(r'\b' + re.escape(aspect) + r'\b', str(text).lower()):
                    window = str(text).lower()[max(0, str(text).lower().find(aspect)-50):min(len(text), str(text).lower().find(aspect)+len(aspect)+50)]
                    polarity = TextBlob(window).sentiment.polarity
                    sentiment_cat = 'Positive' if polarity > 0.1 else 'Negative' if polarity < -0.1 else 'Neutral'
                    all_aspect_sentiments.append({
                        'product_title': product_title,
                        'aspect': aspect, 
                        'sentiment': sentiment_cat
                    })
    
    return pd.DataFrame(all_aspect_sentiments)

def create_aspect_divergent_bar_chart(aspect_df):
    """Creates the divergent stacked bar chart for aspect sentiment."""
    if aspect_df.empty:
        return go.Figure().update_layout(title="No aspect data to display.")

    # Calculate percentages
    summary = aspect_df.groupby(['product_title', 'aspect', 'sentiment']).size().reset_index(name='count')
    total_mentions = summary.groupby(['product_title', 'aspect'])['count'].transform('sum')
    summary['percentage'] = (summary['count'] / total_mentions) * 100

    # Pivot the data for the chart
    pivot_df = summary.pivot_table(index=['product_title', 'aspect'], columns='sentiment', values='percentage', fill_value=0).reset_index()
    
    # Ensure all sentiment columns exist
    for sent in ['Positive', 'Negative', 'Neutral']:
        if sent not in pivot_df.columns:
            pivot_df[sent] = 0
            
    pivot_df['y_axis_label'] = pivot_df['product_title'].str[:25] + "... - " + pivot_df['aspect']
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=pivot_df['y_axis_label'],
        x=pivot_df['Positive'],
        name='Positive', orientation='h', marker_color='#1a9850'
    ))
    fig.add_trace(go.Bar(
        y=pivot_df['y_axis_label'],
        x=-pivot_df['Negative'],  # Negative values for divergence
        name='Negative', orientation='h', marker_color='#d73027'
    ))
    
    fig.update_layout(
        barmode='relative',
        title_text='Comparative Sentiment by Product Aspect (%)',
        xaxis_title='Percentage of Mentions',
        yaxis_title='Product - Aspect',
        yaxis_autorange='reversed',
        plot_bgcolor='white',
        legend_orientation='h',
        legend_yanchor='bottom', legend_y=1.02
    )
    return fig

# (Other helper functions like display_product_header and create_differential_word_clouds remain here)
def display_product_header(product_details, reviews_df):
    with st.container(border=True):
        st.subheader(product_details['product_title'])
        image_url = product_details.get('image_urls', '').split(',')[0] or "https://via.placeholder.com/200"
        st.image(image_url, use_container_width=True)
        st.caption(f"Category: {product_details['category']} | Store: {product_details['store']}")
        m_col1, m_col2 = st.columns(2)
        m_col1.metric("Avg. Rating", f"{product_details.get('average_rating', 0):.2f} ⭐")
        m_col2.metric("Filtered Reviews", f"{len(reviews_df):,}")
        st.markdown("---")
        dist_col1, dist_col2 = st.columns(2)
        with dist_col1:
            st.markdown("**Rating Distribution**")
            if not reviews_df.empty:
                rating_counts = reviews_df['rating'].value_counts().reindex(range(1, 6), fill_value=0)
                for rating in range(5, 0, -1):
                    count = rating_counts.get(rating, 0)
                    percentage = (count / len(reviews_df) * 100) if len(reviews_df) > 0 else 0
                    st.text(f"{rating} ⭐: {percentage:.1f}%")
                    st.progress(int(percentage))
        with dist_col2:
            st.markdown("**Sentiment Distribution**")
            if not reviews_df.empty:
                sentiment_counts = reviews_df['sentiment'].value_counts()
                for sentiment, color in [('Positive', 'green'), ('Neutral', 'grey'), ('Negative', 'red')]:
                    count = sentiment_counts.get(sentiment, 0)
                    percentage = (count / len(reviews_df) * 100) if len(reviews_df) > 0 else 0
                    st.markdown(f":{color}[{sentiment}]: {percentage:.1f}%")
                    st.progress(int(percentage))

def create_differential_word_clouds(review_data_cache, asins):
    if len(asins) < 2: return
    all_texts = [review_data_cache[asin]['text'].str.cat(sep=' ') for asin in asins if not review_data_cache[asin].empty]
    if len(all_texts) < len(asins): return
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    feature_names = vectorizer.get_feature_names_out()
    cols = st.columns(len(asins))
    for i, asin in enumerate(asins):
        with cols[i]:
            product_details = get_product_details(conn, asin).iloc[0]
            st.subheader(f"Unique words for '{product_details['product_title'][:30]}...'")
            scores = {word: score for word, score in zip(feature_names, tfidf_matrix[i].toarray().flatten()) if score > 0}
            if not scores:
                st.info("No unique words found.")
                continue
            wordcloud = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(scores)
            fig, ax = plt.subplots()
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis("off")
            st.pyplot(fig)


# --- Main App Logic ---
def main():
    st.title("⚖️ Advanced Product Comparison")

    if st.button("⬅️ Back to Search"):
        st.switch_page("app.py")

    if 'products_to_compare' not in st.session_state or not st.session_state.products_to_compare:
        st.warning("Please select 2 to 4 products from the main search page to compare.")
        st.stop()
    
    selected_asins = st.session_state.products_to_compare
    if len(selected_asins) < 2:
        st.warning("Please select at least two products to compare.")
        st.stop()

    st.sidebar.header("📊 Universal Comparison Filters")
    min_dates, max_dates = [], []
    for asin in selected_asins:
        min_d, max_d = get_product_date_range(conn, asin)
        min_dates.append(min_d); max_dates.append(max_d)
    
    selected_date_range = st.sidebar.date_input("Filter by Date Range", value=(min(min_dates), max(max_dates)))
    selected_ratings = st.sidebar.multiselect("Filter by Star Rating", options=[1,2,3,4,5], default=[1,2,3,4,5])
    selected_sentiments = st.sidebar.multiselect("Filter by Sentiment", options=['Positive','Negative','Neutral'], default=['Positive','Negative','Neutral'])
    selected_verified = st.sidebar.radio("Filter by Purchase Status", ["All", "Verified Only", "Not Verified"])

    review_data_cache = get_review_data_for_asins(conn, selected_asins, selected_date_range, selected_ratings, selected_sentiments, selected_verified)
    
    st.header("Product Overviews")
    cols = st.columns(len(selected_asins))
    for i, asin in enumerate(selected_asins):
        with cols[i]:
            product_details = get_product_details(conn, asin).iloc[0]
            reviews_df = review_data_cache.get(asin, pd.DataFrame())
            display_product_header(product_details, reviews_df)

    st.markdown("---")
    
    # --- NEW: Interactive Aspect Comparison Tab ---
    tab1, tab2, tab3 = st.tabs(["📊 Aspect Sentiment Comparison", "📈 Time Series", "📝 Differential Text Analysis"])

    with tab1:
        st.subheader("Comparative Aspect-Based Sentiment")
        st.markdown("This chart compares sentiment towards the most common features across all selected products. Use the slider to change the number of features shown.")
        
        num_aspects = st.slider("Select number of aspects to compare:", min_value=3, max_value=15, value=7, key="num_aspects_slider")
        
        aspect_sentiments_df = get_comparative_aspect_sentiments(review_data_cache, num_aspects)
        
        if not aspect_sentiments_df.empty:
            aspect_chart = create_aspect_divergent_bar_chart(aspect_sentiments_df)
            st.plotly_chart(aspect_chart, use_container_width=True, height=max(500, len(aspect_sentiments_df['aspect'].unique()) * 100))
        else:
            st.warning("No common aspects could be found for the selected products and filters.")

    with tab2:
        st.subheader("Sentiment Trends Over Time")
        # This section remains the same
        cols = st.columns(len(selected_asins))
        for i, asin in enumerate(selected_asins):
            with cols[i]:
                product_details = get_product_details(conn, asin).iloc[0]
                st.markdown(f"**Trend for '{product_details['product_title'][:30]}...'**")
                df = review_data_cache.get(asin)
                if df is not None and not df.empty:
                    df['date'] = pd.to_datetime(df['date'])
                    df['period'] = df['date'].dt.to_period('M').dt.start_time
                    sentiment_over_time = df.groupby(['period', 'sentiment']).size().reset_index(name='count')
                    fig = go.Figure()
                    for sentiment, color in [('Positive', '#1a9850'), ('Neutral', '#cccccc'), ('Negative', '#d73027')]:
                        sentiment_df = sentiment_over_time[sentiment_over_time['sentiment'] == sentiment]
                        fig.add_trace(go.Scatter(x=sentiment_df['period'], y=sentiment_df['count'], mode='lines', name=sentiment, stackgroup='one', line_color=color))
                    st.plotly_chart(fig, use_container_width=True, height=300)

    with tab3:
        st.subheader("Differential Word Clouds")
        # This section remains the same
        create_differential_word_clouds(review_data_cache, selected_asins)

if __name__ == "__main__":
    main()
