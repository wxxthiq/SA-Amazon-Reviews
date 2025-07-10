# pages/5_Product_Comparison.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from utils.database_utils import connect_to_db, get_product_details, get_reviews_for_product
from collections import Counter
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Advanced Product Comparison")
DB_PATH = "amazon_reviews_top100.duckdb"
conn = connect_to_db(DB_PATH)

# --- Helper Functions for Advanced Visualizations ---

@st.cache_data
def get_review_data_for_asins(_conn, asins, verified_filter):
    """Fetches and caches review data for a list of ASINs."""
    review_data = {}
    for asin in asins:
        review_data[asin] = get_reviews_for_product(
            _conn, asin, date_range=(), rating_filter=(), sentiment_filter=(), verified_filter=verified_filter
        )
    return review_data

def create_divergent_bar_chart(review_data_cache):
    """Creates a divergent stacked bar chart for sentiment comparison."""
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
                'product': product_title,
                'Positive': pos_pct,
                'Negative': -neg_pct, # Negative values for divergence
                'Neutral': neu_pct,
                'Neutral_Left': -neu_pct / 2, # Split neutral for centering
                'Neutral_Right': neu_pct / 2
            })

    if not plot_data:
        return go.Figure()

    plot_df = pd.DataFrame(plot_data)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(y=plot_df['product'], x=plot_df['Positive'], name='Positive', orientation='h', marker_color='#1a9850'))
    fig.add_trace(go.Bar(y=plot_df['product'], x=plot_df['Negative'], name='Negative', orientation='h', marker_color='#d73027'))
    fig.add_trace(go.Bar(y=plot_df['product'], x=plot_df['Neutral_Right'], name='Neutral', orientation='h', marker_color='#cccccc'))
    fig.add_trace(go.Bar(y=plot_df['product'], x=plot_df['Neutral_Left'], name='Neutral (cont.)', orientation='h', marker_color='#cccccc', showlegend=False))

    fig.update_layout(
        barmode='relative',
        title_text='Comparative Sentiment Distribution (%)',
        xaxis_title='Percentage of Reviews',
        yaxis_title='Product',
        yaxis_autorange='reversed',
        plot_bgcolor='white',
        legend_orientation='h',
        legend_yanchor='bottom',
        legend_y=1.02
    )
    fig.update_xaxes(
        tickvals=[-100, -75, -50, -25, 0, 25, 50, 75, 100],
        ticktext=['100%', '75%', '50%', '25%', '0', '25%', '50%', '75%', '100%']
    )
    return fig

def create_differential_word_clouds(review_data_cache, asins):
    """Calculates unique words for each product and displays them as word clouds."""
    if len(asins) < 2:
        st.warning("Differential analysis requires at least two products.")
        return

    # Combine all review texts into a corpus
    all_texts = [review_data_cache[asin]['text'].str.cat(sep=' ') for asin in asins if not review_data_cache[asin].empty]
    if len(all_texts) != len(asins):
        st.warning("One or more selected products have no review text to analyze.")
        return

    # Use TF-IDF to find characteristic words
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    feature_names = vectorizer.get_feature_names_out()

    cols = st.columns(len(asins))
    for i, asin in enumerate(asins):
        with cols[i]:
            product_details = get_product_details(conn, asin).iloc[0]
            st.subheader(f"Unique words for '{product_details['product_title'][:30]}...'")
            
            # Get the TF-IDF scores for the current product
            scores = tfidf_matrix[i].toarray().flatten()
            
            # Create a dictionary of word -> score
            word_scores = {word: score for word, score in zip(feature_names, scores)}
            
            # Filter out words with zero score
            word_scores = {word: score for word, score in word_scores.items() if score > 0}

            if not word_scores:
                st.info("No unique words could be identified.")
                continue

            wordcloud = WordCloud(
                width=800, height=400, background_color="white", colormap='viridis'
            ).generate_from_frequencies(word_scores)
            
            fig, ax = plt.subplots()
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis("off")
            st.pyplot(fig)

# --- Main App Logic ---
def main():
    st.title("⚖️ Advanced Product Comparison")

    if st.button("⬅️ Back to Search"):
        st.switch_page("app.py")

    # Check for selected products
    if 'products_to_compare' not in st.session_state or not st.session_state.products_to_compare:
        st.warning("Please select 2 to 4 products from the main search page to compare.")
        st.stop()
    
    selected_asins = st.session_state.products_to_compare
    if len(selected_asins) < 2:
        st.warning("Please select at least two products to compare.")
        st.stop()

    st.sidebar.header("📊 Universal Filters")
    verified_filter = st.sidebar.radio(
        "Filter by Purchase Status", ["All", "Verified Only", "Not Verified"], 
        key='comparison_verified_filter'
    )

    # Fetch all data at once
    review_data_cache = get_review_data_for_asins(conn, selected_asins, verified_filter)
    
    # --- Tabbed Interface ---
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Sentiment Overview", "📈 Time Series", "📝 Textual Analysis", "🌐 Advanced Text (Coming Soon)"])

    with tab1:
        st.header("High-Level Sentiment Comparison")
        st.markdown("""
        This chart shows the overall sentiment breakdown for each product. The bars are centered on zero to make it easy 
        to compare the balance between positive and negative feedback.
        """)
        fig = create_divergent_bar_chart(review_data_cache)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.header("Sentiment & Rating Trends Over Time")
        st.markdown("Compare how sentiment for each product has evolved.")
        
        cols = st.columns(len(selected_asins))
        for i, asin in enumerate(selected_asins):
            with cols[i]:
                product_details = get_product_details(conn, asin).iloc[0]
                st.subheader(f"Trend for '{product_details['product_title'][:30]}...'")
                
                df = review_data_cache.get(asin)
                if df is not None and not df.empty:
                    df['date'] = pd.to_datetime(df['date'])
                    df['period'] = df['date'].dt.to_period('M').dt.start_time
                    
                    sentiment_over_time = df.groupby(['period', 'sentiment']).size().reset_index(name='count')
                    
                    fig = go.Figure()
                    for sentiment in ['Positive', 'Neutral', 'Negative']:
                        sentiment_df = sentiment_over_time[sentiment_over_time['sentiment'] == sentiment]
                        fig.add_trace(go.Scatter(
                            x=sentiment_df['period'], y=sentiment_df['count'],
                            hoverinfo='x+y',
                            mode='lines',
                            name=sentiment,
                            stackgroup='one' # This creates the stacked area chart
                        ))
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No data to display trend.")
                    
    with tab3:
        st.header("Differential Textual Analysis")
        st.markdown("""
        These **differential word clouds** highlight words that are uniquely characteristic of each product's reviews 
        when compared to the others. A larger word means it's more representative of that specific product.
        """)
        create_differential_word_clouds(review_data_cache, selected_asins)
        
    with tab4:
        st.header("Deep Textual Comparison")
        st.info("Coming soon: Butterfly charts for direct phrase comparison and side-by-side co-occurrence networks.")


if __name__ == "__main__":
    main()
