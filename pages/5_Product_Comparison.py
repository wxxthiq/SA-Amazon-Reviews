# pages/5_Product_Comparison.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from streamlit_plotly_events import plotly_events
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

REVIEWS_PER_PAGE = 5

@st.cache_resource
def load_spacy_model():
    return spacy.load("en_core_web_sm")
nlp = load_spacy_model()

# --- Session State Initialization for this page ---
if 'aspect_selection' not in st.session_state:
    st.session_state.aspect_selection = None
if 'aspect_review_page' not in st.session_state:
    st.session_state.aspect_review_page = 0

# --- Helper Functions ---
@st.cache_data
def get_review_data_for_asins(_conn, asins, date_range, rating_filter, sentiment_filter, verified_filter):
    review_data = {}
    for asin in asins:
        review_data[asin] = get_reviews_for_product(_conn, asin, date_range, tuple(rating_filter), tuple(sentiment_filter), verified_filter)
    return review_data

@st.cache_data
def get_top_aspects(_review_data_cache, top_n_aspects):
    all_text_corpus = pd.concat([df['text'] for df in _review_data_cache.values() if not df.empty]).astype(str)
    if all_text_corpus.empty: return []
    def clean_chunk(chunk): return " ".join(token.lemma_.lower() for token in chunk if token.pos_ in ['NOUN', 'PROPN', 'ADJ'])
    all_aspects = [clean_chunk(chunk) for doc in nlp.pipe(all_text_corpus) for chunk in doc.noun_chunks if clean_chunk(chunk) and len(clean_chunk(chunk)) > 2]
    if not all_aspects: return []
    return [aspect for aspect, freq in Counter(all_aspects).most_common(top_n_aspects)]

def create_single_product_aspect_chart(product_title, reviews_df, top_aspects):
    """
    --- PUBLICATION-QUALITY VERSION ---
    Creates a true, interactive divergent stacked bar chart with non-rotated,
    readable hover labels.
    """
    aspect_sentiments = []
    for aspect in top_aspects:
        # Find all reviews that mention the aspect
        aspect_reviews = reviews_df[reviews_df['text'].str.contains(r'\b' + re.escape(aspect) + r'\b', case=False, na=False)]
        for _, review in aspect_reviews.iterrows():
            # Use the pre-calculated sentiment for each review
            aspect_sentiments.append({'aspect': aspect, 'sentiment': review['sentiment']})

    if not aspect_sentiments:
        return go.Figure().update_layout(title_text=f"No aspect data for '{product_title[:30]}...'", plot_bgcolor='white')

    aspect_df = pd.DataFrame(aspect_sentiments)
    
    # Calculate counts and percentages
    summary = aspect_df.groupby(['aspect', 'sentiment']).size().unstack(fill_value=0)
    for sent in ['Positive', 'Neutral', 'Negative']:
        if sent not in summary.columns: summary[sent] = 0
            
    summary = summary.reindex(top_aspects).fillna(0)
    summary_pct = summary.div(summary.sum(axis=1), axis=0).fillna(0) * 100

    fig = go.Figure()
    colors = {'Positive': '#1a9850', 'Neutral': '#cccccc', 'Negative': '#d73027'}
    
    # Define traces for each sentiment
    fig.add_trace(go.Bar(
        y=summary_pct.index, x=summary_pct['Positive'], name='Positive', orientation='h',
        marker_color=colors['Positive'], customdata=summary['Positive'],
        hovertemplate="<b>%{y}</b><br>Positive: %{x:.1f}% (%{customdata} mentions)<extra></extra>"
    ))
    fig.add_trace(go.Bar(
        y=summary_pct.index, x=-summary_pct['Negative'], name='Negative', orientation='h',
        marker_color=colors['Negative'], customdata=summary['Negative'],
        hovertemplate="<b>%{y}</b><br>Negative: %{customdata} mentions (%{x:.1f}%)<extra></extra>"
    ))
    fig.add_trace(go.Bar(
        y=summary_pct.index, x=summary_pct['Neutral'], name='Neutral', orientation='h',
        marker_color=colors['Neutral'], customdata=summary['Neutral'],
        base=-summary_pct['Neutral']/2,
        hovertemplate="<b>%{y}</b><br>Neutral: %{customdata} mentions (%{x:.1f}%)<extra></extra>"
    ))

    fig.update_layout(
        barmode='relative',
        title_text=f"Aspect Sentiment for '{product_title[:30]}...'",
        xaxis_title="Percentage of Mentions",
        yaxis_autorange='reversed',
        plot_bgcolor='white',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, traceorder="reversed"),
        height=max(400, len(top_aspects) * 40),
        xaxis=dict(
            tickvals=[-100, -75, -50, -25, 0, 25, 50, 75, 100],
            ticktext=['100%', '75%', '50%', '25%', '0', '25%', '50%', '75%', '100%']
        ),
        # --- NEW: This block ensures hover labels are never rotated ---
        hovermode='y unified',
        hoverlabel=dict(
            bgcolor="white",
            font_size=14,
            font_family="Rockwell"
        )
    )
    return fig
    
# (Other functions like display_product_header and create_differential_word_clouds remain here)
def display_product_header(product_details, reviews_df):
    with st.container(border=True):
        st.subheader(product_details['product_title'])
        image_url = (product_details.get('image_urls') or "").split(',')[0] or "https://via.placeholder.com/200"
        st.image(image_url, use_container_width=True)
        st.caption(f"Category: {product_details['category']} | Store: {product_details.get('store', 'N/A')}")
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
        st.session_state.aspect_selection = None
        st.switch_page("app.py")

    if 'products_to_compare' not in st.session_state or not st.session_state.products_to_compare:
        st.warning("Please select 2 to 4 products from the main page to compare.")
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
    
    def reset_selection(): st.session_state.aspect_selection = None
    selected_date_range = st.sidebar.date_input("Filter by Date Range", value=(min(min_dates), max(max_dates)), on_change=reset_selection)
    selected_ratings = st.sidebar.multiselect("Filter by Star Rating", options=[1,2,3,4,5], default=[1,2,3,4,5], on_change=reset_selection)
    selected_sentiments = st.sidebar.multiselect("Filter by Sentiment", options=['Positive','Negative','Neutral'], default=['Positive','Negative','Neutral'], on_change=reset_selection)
    selected_verified = st.sidebar.radio("Filter by Purchase Status", ["All", "Verified Only", "Not Verified"], on_change=reset_selection)

    review_data_cache = get_review_data_for_asins(conn, selected_asins, selected_date_range, selected_ratings, selected_sentiments, selected_verified)
    
    st.header("Product Overviews")
    cols = st.columns(len(selected_asins))
    for i, asin in enumerate(selected_asins):
        with cols[i]:
            display_product_header(get_product_details(conn, asin).iloc[0], review_data_cache.get(asin, pd.DataFrame()))

    st.markdown("---")
    
    tab1, tab2, tab3 = st.tabs(["📊 Aspect Sentiment Comparison", "📈 Time Series", "📝 Differential Text Analysis"])

    with tab1:
        st.subheader("Comparative Aspect-Based Sentiment")
        st.caption("Click on any bar segment to see the corresponding review snippets below.")
        num_aspects = st.slider("Select number of aspects to compare:", min_value=3, max_value=15, value=7, on_change=reset_selection)
        top_aspects = get_top_aspects(review_data_cache, num_aspects)
        
        if top_aspects:
            chart_cols = st.columns(len(selected_asins))
            for i, asin in enumerate(selected_asins):
                with chart_cols[i]:
                    product_details = get_product_details(conn, asin).iloc[0]
                    reviews_df = review_data_cache.get(asin)
                    if reviews_df is not None and not reviews_df.empty:
                        fig = create_single_product_aspect_chart(product_details['product_title'], reviews_df, top_aspects)
                        selected_point = plotly_events(fig, click_event=True, key=f"aspect_chart_{asin}")
                        if selected_point:
                            st.session_state.aspect_selection = {
                                'asin': asin,
                                'aspect': selected_point[0]['y'],
                                'sentiment': fig.data[selected_point[0]['curveNumber']]['name']
                            }
                            st.session_state.aspect_review_page = 0
                            st.rerun()
                    else:
                        st.info(f"No review data for '{product_details['product_title'][:30]}...' to analyze.")
        else:
            st.warning("No common aspects could be found for the selected products and filters.")
        
        if st.session_state.aspect_selection:
            st.markdown("---")
            selection = st.session_state.aspect_selection
            st.subheader(f"Reviews for '{selection['aspect']}' with '{selection['sentiment']}' sentiment")
            
            product_reviews = review_data_cache.get(selection['asin'])
            filtered_reviews = product_reviews[product_reviews['text'].str.contains(r'\b' + re.escape(selection['aspect']) + r'\b', case=False, na=False)]
            
            if selection['sentiment'] == 'Positive':
                final_reviews = filtered_reviews[filtered_reviews['text_polarity'] > 0.1]
            elif selection['sentiment'] == 'Negative':
                final_reviews = filtered_reviews[filtered_reviews['text_polarity'] < -0.1]
            else: # Neutral
                final_reviews = filtered_reviews[(filtered_reviews['text_polarity'] >= -0.1) & (filtered_reviews['text_polarity'] <= 0.1)]

            if final_reviews.empty:
                st.warning("No matching reviews found.")
            else:
                start_idx = st.session_state.aspect_review_page * REVIEWS_PER_PAGE
                end_idx = start_idx + REVIEWS_PER_PAGE
                reviews_to_display = final_reviews.iloc[start_idx:end_idx]

                for _, review in reviews_to_display.iterrows():
                    with st.container(border=True):
                        st.caption(f"Rating: {review['rating']} ⭐ | Date: {review['date']}")
                        highlighted_text = re.sub(f'({re.escape(selection["aspect"])})', r'<mark>\1</mark>', review['text'], flags=re.IGNORECASE)
                        st.markdown(f"> {highlighted_text}", unsafe_allow_html=True)

                total_reviews = len(final_reviews)
                total_pages = (total_reviews + REVIEWS_PER_PAGE - 1) // REVIEWS_PER_PAGE
                if total_pages > 1:
                    nav_cols = st.columns([1, 1, 1])
                    if st.session_state.aspect_review_page > 0:
                        nav_cols[0].button("⬅️ Previous", on_click=lambda: st.session_state.update(aspect_review_page=st.session_state.aspect_review_page - 1), use_container_width=True)
                    nav_cols[1].write(f"Page {st.session_state.aspect_review_page + 1} of {total_pages}")
                    if (st.session_state.aspect_review_page + 1) < total_pages:
                        nav_cols[2].button("Next ➡️", on_click=lambda: st.session_state.update(aspect_review_page=st.session_state.aspect_review_page + 1), use_container_width=True)

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

    with tab3:
        st.subheader("Differential Word Clouds")
        create_differential_word_clouds(review_data_cache, selected_asins)

if __name__ == "__main__":
    main()
