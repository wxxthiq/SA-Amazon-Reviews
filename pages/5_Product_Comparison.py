# pages/5_Product_Comparison.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from utils.database_utils import connect_to_db, get_product_details, get_reviews_for_product
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
import altair as alt # <-- ADDED IMPORT

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Advanced Product Comparison")
DB_PATH = "amazon_reviews_top100.duckdb"
conn = connect_to_db(DB_PATH)

# --- Helper Functions ---

@st.cache_data
def get_review_data_for_asins(_conn, asins, verified_filter):
    """Fetches and caches review data for a list of ASINs."""
    review_data = {}
    for asin in asins:
        review_data[asin] = get_reviews_for_product(
            _conn, asin, date_range=(), rating_filter=(), sentiment_filter=(), verified_filter=verified_filter
        )
    return review_data

def display_product_metadata_card(product_details, reviews_df):
    """Creates a detailed card for a single product with its metadata and distributions."""
    with st.container(border=True):
        st.subheader(product_details['product_title'])
        image_urls_str = product_details.get('image_urls')
        image_url = image_urls_str.split(',')[0] if pd.notna(image_urls_str) else "https://via.placeholder.com/200"
        st.image(image_url, use_container_width=True)

        m_col1, m_col2 = st.columns(2)
        m_col1.metric("Avg. Rating", f"{product_details.get('average_rating', 0):.2f} ⭐")
        m_col2.metric("Filtered Reviews", f"{len(reviews_df):,}")
        
        st.markdown("---")

        if not reviews_df.empty:
            st.markdown("**Rating Distribution**")
            rating_counts = reviews_df['rating'].value_counts().reindex(range(1, 6), fill_value=0)
            st.bar_chart(rating_counts, height=200)

            # --- FIX: Replaced st.bar_chart with st.altair_chart for sentiment ---
            st.markdown("**Sentiment Distribution**")
            sentiment_df = reviews_df['sentiment'].value_counts().reindex(['Positive', 'Neutral', 'Negative'], fill_value=0).reset_index()
            sentiment_df.columns = ['sentiment', 'count']

            chart = alt.Chart(sentiment_df).mark_bar().encode(
                x=alt.X('sentiment', sort=['Positive', 'Neutral', 'Negative'], title=None, axis=alt.Axis(labels=False)),
                y=alt.Y('count', title=None, axis=alt.Axis(labels=False)),
                color=alt.Color('sentiment',
                                scale=alt.Scale(
                                    domain=['Positive', 'Neutral', 'Negative'],
                                    range=['#1a9850', '#cccccc', '#d73027']
                                ),
                                legend=alt.Legend(title="Sentiment", orient="bottom")),
                tooltip=['sentiment', 'count']
            ).properties(height=200).configure_view(strokeWidth=0)
            
            st.altair_chart(chart, use_container_width=True)
            
        else:
            st.info("No review data to display distributions.")

def create_divergent_bar_chart(review_data_cache):
    # This function remains the same
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
    # This function remains the same
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

    st.sidebar.header("📊 Universal Filters")
    verified_filter = st.sidebar.radio("Filter by Purchase Status", ["All", "Verified Only", "Not Verified"], key='comparison_verified_filter')

    review_data_cache = get_review_data_for_asins(conn, selected_asins, verified_filter)
    
    st.header("Product Overviews")
    cols = st.columns(len(selected_asins))
    for i, asin in enumerate(selected_asins):
        with cols[i]:
            product_details = get_product_details(conn, asin).iloc[0]
            reviews_df = review_data_cache.get(asin, pd.DataFrame())
            display_product_metadata_card(product_details, reviews_df)

    st.markdown("---")
    
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
                    df['period'] = df['date'].dt.to_period('M').dt.start_time
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
