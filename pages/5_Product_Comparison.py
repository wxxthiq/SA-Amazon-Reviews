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

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Advanced Product Comparison")
DB_PATH = "amazon_reviews_top100.duckdb"
conn = connect_to_db(DB_PATH)

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

def display_product_header(product_details, reviews_df):
    """
    Creates a detailed header for a single product, matching the Sentiment Overview page.
    """
    with st.container(border=True):
        # --- Product Title and Image ---
        st.subheader(product_details['product_title'])
        image_urls_str = product_details.get('image_urls')
        image_url = image_urls_str.split(',')[0] if pd.notna(image_urls_str) else "https://via.placeholder.com/200"
        st.image(image_url, use_container_width=True)
        st.caption(f"Category: {product_details['category']} | Store: {product_details['store']}")

        # --- Key Metrics ---
        m_col1, m_col2 = st.columns(2)
        m_col1.metric("Avg. Rating", f"{product_details.get('average_rating', 0):.2f} ⭐")
        m_col2.metric("Filtered Reviews", f"{len(reviews_df):,}")
        
        st.markdown("---")

        # --- Distribution Charts ---
        dist_col1, dist_col2 = st.columns(2)
        with dist_col1:
            st.markdown("**Rating Distribution**")
            if not reviews_df.empty:
                rating_counts = reviews_df['rating'].value_counts().reindex(range(1, 6), fill_value=0)
                total_ratings = len(reviews_df)
                for rating in range(5, 0, -1):
                    count = rating_counts.get(rating, 0)
                    percentage = (count / total_ratings * 100) if total_ratings > 0 else 0
                    st.text(f"{rating} ⭐: {percentage:.1f}% ({count})")
                    st.progress(int(percentage))
            else:
                st.info("No ratings.")

        with dist_col2:
            st.markdown("**Sentiment Distribution**")
            if not reviews_df.empty:
                sentiment_counts = reviews_df['sentiment'].value_counts()
                total_sentiments = len(reviews_df)
                sentiment_colors = {"Positive": "green", "Neutral": "grey", "Negative": "red"}
                for sentiment in ['Positive', 'Neutral', 'Negative']:
                    count = sentiment_counts.get(sentiment, 0)
                    percentage = (count / total_sentiments * 100) if total_sentiments > 0 else 0
                    st.markdown(f":{sentiment_colors.get(sentiment, 'default')}[{sentiment}]: {percentage:.1f}% ({count})")
                    st.progress(int(percentage))
            else:
                st.info("No sentiments.")

# (Other helper functions like create_divergent_bar_chart and create_differential_word_clouds remain the same)
def create_divergent_bar_chart(review_data_cache):
    plot_data = []
    for asin, df in review_data_cache.items():
        if not df.empty:
            total = len(df)
            counts = df['sentiment'].value_counts()
            pos_pct = counts.get('Positive', 0) / total * 100
            neg_pct = counts.get('Negative', 0) / total * 100
            neu_pct = counts.get('Neutral', 0) / total * 100
            product_details = get_product_details(conn, asin).iloc[0]
            product_title = product_details['product_title']
            plot_data.append({
                'product': product_title, 'Positive': pos_pct, 'Negative': -neg_pct,
                'Neutral_Left': -neu_pct / 2, 'Neutral_Right': neu_pct / 2
            })
    if not plot_data: return go.Figure()
    plot_df = pd.DataFrame(plot_data)
    fig = go.Figure()
    fig.add_trace(go.Bar(y=plot_df['product'], x=plot_df['Positive'], name='Positive', orientation='h', marker_color='#1a9850'))
    fig.add_trace(go.Bar(y=plot_df['product'], x=plot_df['Negative'], name='Negative', orientation='h', marker_color='#d73027'))
    fig.add_trace(go.Bar(y=plot_df['product'], x=plot_df['Neutral_Right'], name='Neutral', orientation='h', marker_color='#cccccc'))
    fig.add_trace(go.Bar(y=plot_df['product'], x=plot_df['Neutral_Left'], showlegend=False, orientation='h', marker_color='#cccccc'))
    fig.update_layout(barmode='relative', title_text='Comparative Sentiment Distribution (%)', yaxis_autorange='reversed')
    return fig

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

    # --- COMPLETE Universal Sidebar Filters ---
    st.sidebar.header("📊 Universal Comparison Filters")
    
    # Get combined date range for all selected products
    min_dates, max_dates = [], []
    for asin in selected_asins:
        min_d, max_d = get_product_date_range(conn, asin)
        min_dates.append(min_d)
        max_dates.append(max_d)
    
    overall_min_date = min(min_dates) if min_dates else datetime.now().date()
    overall_max_date = max(max_dates) if max_dates else datetime.now().date()

    default_date_range = (overall_min_date, overall_max_date)
    default_ratings = [1, 2, 3, 4, 5]
    default_sentiments = ['Positive', 'Negative', 'Neutral']
    
    selected_date_range = st.sidebar.date_input("Filter by Date Range", value=default_date_range, key='comp_date_filter')
    selected_ratings = st.sidebar.multiselect("Filter by Star Rating", options=default_ratings, default=default_ratings, key='comp_rating_filter')
    selected_sentiments = st.sidebar.multiselect("Filter by Sentiment", options=default_sentiments, default=default_sentiments, key='comp_sentiment_filter')
    selected_verified = st.sidebar.radio("Filter by Purchase Status", ["All", "Verified Only", "Not Verified"], index=0, key='comp_verified_filter')

    # --- Data Fetching with ALL filters ---
    review_data_cache = get_review_data_for_asins(conn, selected_asins, selected_date_range, selected_ratings, selected_sentiments, selected_verified)
    
    # --- Display Product Headers ---
    st.header("Product Overviews")
    st.caption("The following overviews reflect the universal filters selected in the sidebar.")
    cols = st.columns(len(selected_asins))
    for i, asin in enumerate(selected_asins):
        with cols[i]:
            product_details = get_product_details(conn, asin).iloc[0]
            reviews_df = review_data_cache.get(asin, pd.DataFrame())
            display_product_header(product_details, reviews_df)

    st.markdown("---")
    
    # --- Tabbed Interface for Advanced Visualizations ---
    tab1, tab2, tab3 = st.tabs(["📊 Divergent Sentiment Chart", "📈 Time Series", "📝 Differential Text Analysis"])

    with tab1:
        st.subheader("High-Level Sentiment Comparison")
        fig = create_divergent_bar_chart(review_data_cache)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Sentiment Trends Over Time")
        cols = st.columns(len(selected_asins))
        for i, asin in enumerate(selected_asins):
            with cols[i]:
                product_details = get_product_details(conn, asin).iloc[0]
                st.markdown(f"**Trend for '{product_details['product_title'][:30]}...'**")
                df = review_data_cache.get(asin)
                if df is not None and not df.empty:
                    df['date'] = pd.to_datetime(df['date'])
                    time_granularity = 'M' # Monthly
                    df['period'] = df['date'].dt.to_period(time_granularity).dt.start_time
                    sentiment_over_time = df.groupby(['period', 'sentiment']).size().reset_index(name='count')
                    fig = go.Figure()
                    for sentiment, color in [('Positive', '#1a9850'), ('Neutral', '#cccccc'), ('Negative', '#d73027')]:
                        sentiment_df = sentiment_over_time[sentiment_over_time['sentiment'] == sentiment]
                        fig.add_trace(go.Scatter(x=sentiment_df['period'], y=sentiment_df['count'], mode='lines', name=sentiment, stackgroup='one', line_color=color))
                    st.plotly_chart(fig, use_container_width=True, height=300)
                else:
                    st.info("No data to display trend.")
                    
    with tab3:
        st.subheader("Differential Word Clouds")
        create_differential_word_clouds(review_data_cache, selected_asins)

if __name__ == "__main__":
    main()
