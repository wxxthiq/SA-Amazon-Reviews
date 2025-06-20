import streamlit as st
import pandas as pd
import sqlite3
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import altair as alt
import re
from collections import Counter
# --- New Imports for Advanced Analysis ---
import spacy
from textblob import TextBlob

# --- ADD THIS LINE AT THE TOP OF YOUR SCRIPT ---
# This is a best practice that prevents errors if a product has > 5000 reviews
alt.data_transformers.enable('default', max_rows=None)

# --- Configuration ---
DATABASE_PATH = "/kaggle/input/db-fashion-reviews/amazon_reviews.db"
PRODUCTS_PER_PAGE = 16

# --- Database Connection ---
@st.cache_resource
def connect_to_db(path):
    """Creates a connection to the SQLite database."""
    try:
        return sqlite3.connect(path, uri=True, check_same_thread=False)
    except Exception as e:
        st.error(f"FATAL: Could not connect to the database. Error: {e}")
        return None

# --- NLP Model Loading ---
@st.cache_resource
def load_spacy_model():
    """Loads the spaCy model."""
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        st.error("spaCy model 'en_core_web_sm' not found. Please run 'python -m spacy download en_core_web_sm'")
        return None

# --- Data Fetching Functions ---
@st.cache_data
def load_gallery_data(_conn):
    """Loads the unique product data for the main gallery view."""
    if _conn is None: return None
    try:
        table_check_query = "SELECT name FROM sqlite_master WHERE type='table' AND name='reviews';"
        tables = pd.read_sql_query(table_check_query, _conn)
        if tables.empty:
            st.error("DATABASE ERROR: The 'reviews' table was not found. Please check your database file.")
            return None
        
        query = "SELECT parent_asin, product_title, image_url, average_rating, review_count, category FROM reviews GROUP BY parent_asin"
        df = pd.read_sql_query(query, _conn)
        return df
    except Exception as e:
        st.error(f"An error occurred while loading gallery data: {e}")
        return None

def search_products(_conn, category, search_term, sort_by, page=0):
    """Searches and paginates products from the database."""
    if _conn is None: return pd.DataFrame()
    offset = page * PRODUCTS_PER_PAGE
    query = "SELECT parent_asin, product_title, image_url, average_rating, review_count, category FROM reviews"
    
    conditions, params = [], []
    if category != "All":
        conditions.append("category = ?")
        params.append(category)
    if search_term:
        conditions.append("product_title LIKE ?")
        params.append(f"%{search_term}%")
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
        
    query += " GROUP BY parent_asin"
    
    if sort_by == "Popularity (Most Reviews)":
        query += " ORDER BY review_count DESC"
    elif sort_by == "Highest Rating":
        query += " ORDER BY average_rating DESC"
    elif sort_by == "Lowest Rating":
        query += " ORDER BY average_rating ASC"
        
    query += f" LIMIT {PRODUCTS_PER_PAGE} OFFSET {offset}"
    
    try:
        df = pd.read_sql_query(query, _conn, params=params)
        return df
    except Exception as e:
        st.error(f"An error occurred while searching: {e}")
        return pd.DataFrame()

def fetch_reviews_for_product(_conn, asin):
    """Fetches all reviews for a single product."""
    if _conn is None: return pd.DataFrame()
    query = "SELECT * FROM reviews WHERE parent_asin = ?"
    df = pd.read_sql_query(query, _conn, params=(asin,))
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

    # Add a unique ID to each review for interactivity
    df = df.reset_index().rename(columns={'index': 'review_id'})
    
    return df

# --- Visualization & Analysis Functions ---
def generate_wordcloud(text, title, custom_stopwords):
    stopwords = STOPWORDS.union(set(custom_stopwords))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title(title, fontsize=16)
    if text and isinstance(text, str) and text.strip():
        try:
            wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=stopwords, colormap='viridis').generate(text)
            ax.imshow(wordcloud, interpolation='bilinear')
        except ValueError:
            ax.text(0.5, 0.5, 'No words to display', ha='center', va='center')
    else:
        ax.text(0.5, 0.5, 'No reviews for this data segment', ha='center', va='center')
    ax.axis('off')
    return fig

def get_top_keywords(text, custom_stopwords, top_n=50):
    if not isinstance(text, str): return []
    stopwords = STOPWORDS.union(set(custom_stopwords))
    words = re.findall(r"[a-zA-Z']+", text.lower())
    filtered_words = [word for word in words if word not in stopwords and len(word) > 2]
    word_counts = Counter(filtered_words)
    return [word for word, count in word_counts.most_common(top_n)]

# Replace the old calculate_discrepancy function with this new version

# In your app.py, replace the old function with this one.
# The new line is commented.

@st.cache_data
def calculate_discrepancy(reviews_df):
    """Calculates sentiment polarity and discrepancy from star rating."""
    df = reviews_df.copy()
    df.dropna(subset=['text'], inplace=True)

    # --- NEW: Add this line for data type safety ---
    # Force the rating column to be a numeric type, just in case.
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    df.dropna(subset=['rating'], inplace=True) # Drop rows where rating might be non-numeric

    # Use the original, precise rating for accurate calculations
    df['normalized_rating'] = (df['rating'] - 3) / 2
    df['text_polarity'] = df['text'].astype(str).apply(lambda x: TextBlob(x).sentiment.polarity)
    df['discrepancy'] = (df['text_polarity'] - df['normalized_rating']).abs()

    # Create the display column for categorization
    df['rating_display'] = df['rating'].round().astype(int)
    
    return df

@st.cache_data
def aspect_based_sentiment(_reviews_df, aspects, _nlp_model):
    """Performs basic aspect-based sentiment analysis."""
    aspect_sentiments = {aspect: {'pos': 0, 'neg': 0, 'neu': 0, 'mentions': 0} for aspect in aspects}
    
    for review_text in _reviews_df['text'].dropna():
        for sentence in _nlp_model(review_text).sents:
            for aspect in aspects:
                if f' {aspect} ' in sentence.text.lower():
                    aspect_sentiments[aspect]['mentions'] += 1
                    sentiment = TextBlob(sentence.text).sentiment.polarity
                    if sentiment > 0.1:
                        aspect_sentiments[aspect]['pos'] += 1
                    elif sentiment < -0.1:
                        aspect_sentiments[aspect]['neg'] += 1
                    else:
                        aspect_sentiments[aspect]['neu'] += 1
                            
    results = [{'aspect': aspect, **scores} for aspect, scores in aspect_sentiments.items() if scores['mentions'] > 0]
    return pd.DataFrame(results)

# --- Main App ---
st.set_page_config(layout="wide", page_title="Amazon Review Explorer")
st.title("⚡ Test Advanced Sentiment Dashboard")

conn = connect_to_db(DATABASE_PATH)
nlp = load_spacy_model()

if 'page' not in st.session_state: st.session_state.page = 0
if 'selected_product' not in st.session_state: st.session_state.selected_product = None
if 'search_results' not in st.session_state: st.session_state.search_results = pd.DataFrame()
if 'search_clicked' not in st.session_state: st.session_state.search_clicked = False

products_df = load_gallery_data(conn)

# In your app.py, replace the entire "DETAILED PRODUCT VIEW" block with this corrected version.
# The key changes are commented below.

# In your app.py, replace the entire "DETAILED PRODUCT VIEW" block with this corrected version.
# The key changes are commented below.

# In your app.py, replace the entire "DETAILED PRODUCT VIEW" block with this final, corrected version.

# In your app.py, replace the entire "DETAILED PRODUCT VIEW" block with this final, corrected version.

# --- DETAILED PRODUCT VIEW ---
if st.session_state.selected_product:
    selected_asin = st.session_state.selected_product
    if products_df is None:
        st.error("Product data could not be loaded. Please refresh.")
        st.stop()
        
    product_reviews = fetch_reviews_for_product(conn, selected_asin)
    product_details_series = products_df[products_df['parent_asin'] == selected_asin]
    
    if product_details_series.empty:
        st.error(f"Could not find details for product ASIN: {selected_asin}")
        if st.button("⬅️ Back to Search"): st.session_state.selected_product = None
        st.stop()
    else:
        product_details = product_details_series.iloc[0]

    if st.button("⬅️ Back to Search Results"):
        st.session_state.selected_product = None
        st.rerun()

    if not product_reviews.empty and nlp is not None:
        # --- Product Header ---
        col1, col2 = st.columns([1, 4])
        with col1:
            st.image(product_details['image_url'], use_container_width=True)
        with col2:
            st.header(product_details['product_title'])
            stat_cols = st.columns(3)
            stat_cols[0].metric("Average Rating", f"{product_details['average_rating']:.2f} ⭐")
            stat_cols[1].metric("Total Reviews", f"{int(product_details['review_count'])}")
            stat_cols[2].metric("Category", product_details['category'])
        st.markdown("---")
        
        # --- Perform advanced analysis ---
        with st.spinner("Running advanced analysis on reviews..."):
            reviews_with_scores = calculate_discrepancy(product_reviews)
            fashion_aspects = ['fit', 'size', 'color', 'fabric', 'quality', 'price', 'style', 'comfort', 'stitching', 'zipper', 'shipping']
            absa_results_df = aspect_based_sentiment(reviews_with_scores, fashion_aspects, nlp)

        # --- NEW TABBED LAYOUT ---
        overview_tab, keyword_tab = st.tabs(["📊 Comprehensive Analysis", "🔎 Keyword Explorer"])

        with overview_tab:
            st.subheader("Deep Dive into Review Sentiment")

            # --- Chart 1: Aspect Sentiment Chart (Unchanged and correct) ---
            st.markdown("#### Aspect Sentiment")
            st.info("What specific features do customers mention positively or negatively?", icon="💡")
            if not absa_results_df.empty:
                absa_melted = absa_results_df.melt(id_vars=['aspect'], value_vars=['pos', 'neg'], var_name='sentiment', value_name='count')
                absa_melted['count'] = absa_melted.apply(lambda row: -row['count'] if row['sentiment'] == 'neg' else row['count'], axis=1)
                
                absa_chart = alt.Chart(absa_melted).mark_bar().encode(
                    x=alt.X('count:Q', title='Number of Mentions'),
                    y=alt.Y('aspect:N', sort='-x', title='Product Aspect'),
                    color=alt.Color('sentiment:N', scale=alt.Scale(domain=['pos', 'neg'], range=['#2ca02c', '#d62728']), legend=alt.Legend(title="Sentiment")),
                    tooltip=[alt.Tooltip('aspect:N'), alt.Tooltip('count:Q', title='Mentions')]
                ).properties(title="Positive vs. Negative Mentions by Feature")
                st.altair_chart(absa_chart, use_container_width=True)
            else:
                st.warning("No defined aspects found in the reviews for this product.")
            
            st.markdown("---")

            # --- FIX: REPLACING THE ALTAIR CHART WITH A PLOTLY CHART ---
            st.markdown("#### Rating vs. Text Discrepancy")
            st.info("Hover over points to see review snippets. Select a highly discrepant review from the dropdown below to read the full text.", icon="💡")

            # Import plotly express
            import plotly.express as px

            # Create the scatter plot using Plotly Express
            discrepancy_plot = px.scatter(
                reviews_with_scores,
                x="rating_display",
                y="text_polarity",
                color="discrepancy",
                color_continuous_scale=px.colors.sequential.Viridis,
                hover_data=['review_id', 'text'], # Show text snippet on hover
                labels={
                    "rating_display": "Star Rating (Rounded)",
                    "text_polarity": "Sentiment of Review Text",
                    "discrepancy": "Discrepancy Score"
                },
                title="Brighter points indicate higher discrepancy"
            )

            # Make the x-axis categorical to ensure distinct columns
            discrepancy_plot.update_xaxes(type='category', categoryorder='category ascending')
            discrepancy_plot.update_layout(height=400)
            
            # Display the Plotly chart
            st.plotly_chart(discrepancy_plot, use_container_width=True)

            # --- New, Robust Interaction Method ---
            st.markdown("#### Read a Highly Discrepant Review")
            
            # Get the top 10 most discrepant reviews
            top_discrepant_reviews = reviews_with_scores.sort_values('discrepancy', ascending=False).head(10)
            
            # Create a list of options for the selectbox
            options = []
            for _, row in top_discrepant_reviews.iterrows():
                # Format the option label like: "ID: 123 | Rating: 1 | Discrepancy: 1.54"
                option_label = f"ID: {row['review_id']} | Rating: {row['rating']} | Discrepancy: {row['discrepancy']:.2f}"
                options.append(option_label)
            
            # Create the selectbox
            selected_review_label = st.selectbox("Select one of the top 10 most discrepant reviews to read:", options=options)
            
            # Display the full text of the selected review
            if selected_review_label:
                # Extract the ID from the selected label
                selected_id = int(selected_review_label.split('|')[0].replace('ID:', '').strip())
                full_text = reviews_with_scores[reviews_with_scores['review_id'] == selected_id]['text'].iloc[0]
                with st.container(border=True, height=200):
                    st.write(full_text)

            st.markdown("---")
            st.markdown("#### Sentiment Over Time")
            streamgraph = alt.Chart(product_reviews).mark_area().encode(
                x=alt.X('yearmonth(timestamp):T', title='Month'),
                y=alt.Y('count():Q', stack='center', title='Review Volume'),
                color=alt.Color('sentiment:N', scale=alt.Scale(domain=['Positive', 'Neutral', 'Negative'], range=['#2ca02c', '#ff7f0e', '#d62728']), title='Sentiment'),
                tooltip=[alt.Tooltip('yearmonth(timestamp):T', title='Month'), alt.Tooltip('sentiment:N'), alt.Tooltip('count():Q', title='Reviews')]
            ).properties(title="How sentiment has trended over the product's lifetime").interactive()
            st.altair_chart(streamgraph, use_container_width=True)


        with keyword_tab:
            # The code for the keyword_tab remains unchanged and is correct
            st.subheader("Explore Keywords in Reviews")
            st.info("Use the filters to generate a word cloud and explore snippets from specific reviews.", icon="💡")
            filter_col, wordcloud_col = st.columns([1, 2], gap="large")
            with filter_col:
                 st.markdown("#### Filters")
                 with st.container(border=True):
                    all_sentiments = ['Positive', 'Neutral', 'Negative']
                    selected_sentiments = st.multiselect("Filter by Sentiment", options=all_sentiments, default=all_sentiments)
                    valid_timestamps = product_reviews['timestamp'].dropna()
                    if not valid_timestamps.empty:
                        min_date = valid_timestamps.min().date()
                        max_date = valid_timestamps.max().date()
                        selected_date_range = st.date_input("Filter by Date", value=(min_date, max_date), min_value=min_date, max_value=max_date)
                    else:
                        selected_date_range = []
                    exclude_words_input = st.text_input("Exclude words (comma-separated):", placeholder="e.g., product, item", key=f"exclude_input_{selected_asin}")
            dashboard_filtered_reviews = product_reviews[product_reviews['sentiment'].isin(selected_sentiments)]
            if len(selected_date_range) == 2:
                dashboard_filtered_reviews = dashboard_filtered_reviews[
                    (dashboard_filtered_reviews['timestamp'].dt.date >= selected_date_range[0]) &
                    (dashboard_filtered_reviews['timestamp'].dt.date <= selected_date_range[1])
                ]
            with wordcloud_col:
                excluded_words = [word.strip().lower() for word in exclude_words_input.split(',')]
                wordcloud_text = ' '.join(dashboard_filtered_reviews['text'].dropna())
                if not dashboard_filtered_reviews.empty and wordcloud_text.strip():
                    wordcloud_fig = generate_wordcloud(wordcloud_text, "Top Keywords in Selection", excluded_words)
                    st.pyplot(wordcloud_fig)
                else:
                    st.warning("No reviews match criteria to generate a word cloud.")
            st.markdown("---") 
            with st.expander("Explore Keywords and Review Snippets from your selection", expanded=True):
                if not dashboard_filtered_reviews.empty and wordcloud_text.strip():
                    top_keywords = get_top_keywords(wordcloud_text, excluded_words)
                    if top_keywords:
                        top_keywords.insert(0, "--- Select a keyword ---")
                        selected_keyword = st.selectbox("Select a keyword:", options=top_keywords, key="sb_explore")
                        if selected_keyword != "--- Select a keyword ---":
                            snippet_count_key = f"snippet_count_{selected_keyword}"
                            if snippet_count_key not in st.session_state:
                                st.session_state[snippet_count_key] = 3
                            st.markdown(f"**Exploring '{selected_keyword}'**")
                            reviews_with_keyword = dashboard_filtered_reviews[dashboard_filtered_reviews['text'].str.lower().str.contains(f'\\b{re.escape(selected_keyword)}\\b', regex=True, na=False)]
                            st.markdown(f"Found **{len(reviews_with_keyword)}** mentions in your selection.")
                            if not reviews_with_keyword.empty:
                                st.markdown(f"Average rating of these reviews: **{reviews_with_keyword['rating'].mean():.2f}** ⭐")
                                st.markdown("**Review Snippets:**")
                                for _, review in reviews_with_keyword.head(st.session_state[snippet_count_key]).iterrows():
                                    highlighted_text = re.sub(f'({re.escape(selected_keyword)})', r'**\1**', review['text'], flags=re.IGNORECASE)
                                    with st.container(border=True):
                                        st.write(f"**Rating: {review['rating']} ⭐ | Sentiment: {review['sentiment']}**")
                                        st.write(f"...{highlighted_text}...")
                                if len(reviews_with_keyword) > st.session_state[snippet_count_key]:
                                    if st.button("Load more snippets", key=f"load_more_{selected_keyword}"):
                                        st.session_state[snippet_count_key] += 5
                                        st.rerun()
                    else:
                        st.caption("No unique keywords to explore.")
                else:
                    st.caption("No reviews to analyze for keywords.")
    else:
        st.error("Could not find review data for this product or the NLP model failed to load.")
# --- FIX: RESTORED THE MISSING ELSE BLOCK FOR THE MAIN SEARCH PAGE ---
else:
    st.header("Search for Products")
    col1, col2, col3 = st.columns(3)
    with col1:
        search_term = st.text_input("Search by product title:")
    with col2:
        if products_df is not None:
            available_categories = ["All"] + sorted(products_df['category'].unique().tolist())
        else:
            available_categories = ["All"]
        category = st.selectbox("Filter by Category", available_categories)
    with col3:
        sort_by = st.selectbox("Sort By", ["Popularity (Most Reviews)", "Highest Rating", "Lowest Rating"])
    
    if st.button("Search", type="primary"):
        st.session_state.page = 0
        st.session_state.search_clicked = True
        with st.spinner("Searching..."):
            st.session_state.search_results = search_products(conn, category, search_term, sort_by, page=0)

    if not st.session_state.search_results.empty:
        st.markdown("---")
        st.header("Search Results")
        for i in range(0, len(st.session_state.search_results), 4):
            cols = st.columns(4)
            for j in range(4):
                if i + j < len(st.session_state.search_results):
                    row = st.session_state.search_results.iloc[i+j]
                    with cols[j].container(border=True):
                        st.image(row['image_url'], use_container_width=True)
                        st.markdown(f"**{row['product_title']}**")
                        st.caption(f"Avg. Rating: {row['average_rating']:.2f} ⭐ ({int(row['review_count'])} reviews)")
                        if st.button("View Details", key=row['parent_asin']):
                            st.session_state.selected_product = row['parent_asin']
                            st.rerun()
        st.markdown("---")
        
        col_center, _ = st.columns([1, 3])
        with col_center:
            if len(st.session_state.search_results) % PRODUCTS_PER_PAGE == 0 and len(st.session_state.search_results) > 0:
                if st.button("Load More Results"):
                    st.session_state.page += 1
                    with st.spinner("Loading more..."):
                        new_results = search_products(conn, category, search_term, sort_by, page=st.session_state.page)
                        if not new_results.empty:
                            st.session_state.search_results = pd.concat([st.session_state.search_results, new_results], ignore_index=True)
                        else:
                            st.toast("No more results found.")
                    st.rerun()

    elif st.session_state.get('search_clicked'):
        st.info("No products found matching your criteria.")
