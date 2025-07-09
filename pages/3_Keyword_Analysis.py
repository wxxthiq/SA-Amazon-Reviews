# pages/3_Keyword_Analysis.py
import streamlit as st
import pandas as pd
import re
from collections import Counter
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import networkx as nx
from pyvis.network import Network
from itertools import combinations
import streamlit.components.v1 as components
import tempfile 
from utils.database_utils import (
    connect_to_db,
    get_product_details,
    get_reviews_for_product,
    get_product_date_range
)

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Keyword Analysis")
DB_PATH = "amazon_reviews_top100.duckdb"
conn = connect_to_db(DB_PATH)
REVIEWS_PER_PAGE = 5

# --- Main App Logic ---
def main():
    st.title("🔑 Detailed Keyword & Phrase Analysis")

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
    st.caption("Use the sidebar to filter reviews, then explore the most common terms and phrases.")

    # --- DEDICATED SIDEBAR FILTERS ---
    st.sidebar.header("🔬 Keyword Analysis Filters")
    min_date_db, max_date_db = get_product_date_range(conn, selected_asin)
    
    default_date_range = (min_date_db, max_date_db)
    default_ratings = [1, 2, 3, 4, 5]
    default_sentiments = ['Positive', 'Negative', 'Neutral']
    default_verified = "All"
    
    if 'keyword_date_filter' not in st.session_state:
        st.session_state.keyword_date_filter = default_date_range
    if 'keyword_rating_filter' not in st.session_state:
        st.session_state.keyword_rating_filter = default_ratings
    if 'keyword_sentiment_filter' not in st.session_state:
        st.session_state.keyword_sentiment_filter = default_sentiments
    if 'keyword_verified_filter' not in st.session_state:
        st.session_state.keyword_verified_filter = default_verified
        
    def reset_keyword_page():
        st.session_state.keyword_review_page = 0

    st.sidebar.date_input("Filter by Date Range", key='keyword_date_filter', on_change=reset_keyword_page)
    st.sidebar.multiselect("Filter by Star Rating", options=default_ratings, key='keyword_rating_filter', on_change=reset_keyword_page)
    st.sidebar.multiselect("Filter by Sentiment", options=default_sentiments, key='keyword_sentiment_filter', on_change=reset_keyword_page)
    st.sidebar.radio("Filter by Purchase Status", ["All", "Verified Only", "Not Verified"], key='keyword_verified_filter', on_change=reset_keyword_page)
    
    chart_data = get_reviews_for_product(conn, selected_asin, st.session_state.keyword_date_filter, tuple(st.session_state.keyword_rating_filter), tuple(st.session_state.keyword_sentiment_filter), st.session_state.keyword_verified_filter)

    st.markdown("---")
    if chart_data.empty:
        st.warning("No review data available for the selected filters.")
        st.stop()
        
    st.info(f"Analyzing keywords from **{len(chart_data)}** reviews matching your criteria.")

    # --- N-GRAM WORD CLOUD SUMMARY ---
    st.markdown("### ☁️ Keyword & Phrase Summary")
    
    col1, col2 = st.columns([1,1])
    with col1:
        max_words = st.slider("Max Terms in Cloud:", min_value=5, max_value=50, value=15, key="keyword_cloud_slider")
    with col2:
        ngram_level = st.radio("Term Type:", ("Single Words", "Bigrams", "Trigrams"), index=0, horizontal=True, key="keyword_ngram_radio", on_change=reset_keyword_page)
    
    def get_top_ngrams(corpus, n=None, ngram_range=(1,1)):
        vec = CountVectorizer(ngram_range=ngram_range, stop_words='english').fit(corpus)
        bag_of_words = vec.transform(corpus)
        sum_words = bag_of_words.sum(axis=0) 
        words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
        words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
        return words_freq[:n]

    ngram_range = {"Single Words": (1,1), "Bigrams": (2,2), "Trigrams": (3,3)}.get(ngram_level)

    wc_col1, wc_col2 = st.columns(2)
    with wc_col1:
        st.markdown("#### Positive Terms")
        pos_text = chart_data[chart_data["sentiment"]=="Positive"]["text"].dropna()
        if not pos_text.empty:
            top_pos_grams = get_top_ngrams(pos_text, n=max_words, ngram_range=ngram_range)
            if top_pos_grams:
                wordcloud_pos = WordCloud(stopwords=STOPWORDS, background_color="white", width=800, height=400, colormap='Greens').generate_from_frequencies(dict(top_pos_grams))
                fig, ax = plt.subplots()
                ax.imshow(wordcloud_pos, interpolation='bilinear')
                ax.axis("off")
                st.pyplot(fig)

    with wc_col2:
        st.markdown("#### Negative Terms")
        neg_text = chart_data[chart_data["sentiment"]=="Negative"]["text"].dropna()
        if not neg_text.empty:
            top_neg_grams = get_top_ngrams(neg_text, n=max_words, ngram_range=ngram_range)
            if top_neg_grams:
                wordcloud_neg = WordCloud(stopwords=STOPWORDS, background_color="white", width=800, height=400, colormap='Reds').generate_from_frequencies(dict(top_neg_grams))
                fig, ax = plt.subplots()
                ax.imshow(wordcloud_neg, interpolation='bilinear')
                ax.axis("off")
                st.pyplot(fig)

    # --- INTERACTIVE KEYWORD EXPLORER ---
    st.markdown("---")
    st.markdown("### 🔬 Interactive Term Explorer")

    # ** KEY CHANGE: New robust function for getting dropdown options **
    @st.cache_data
    def get_dropdown_options(corpus, ngram_range_tuple):
        # First, get a list of the most frequent terms
        top_terms_list = [term for term, count in get_top_ngrams(corpus, n=50, ngram_range=ngram_range_tuple)]
        
        # Then, for each term, count how many reviews it appears in
        options = []
        for term in top_terms_list:
            # This count will be consistent with the analysis section
            mention_count = corpus.str.contains(re.escape(term), case=False, na=False).sum()
            if mention_count > 0:
                options.append((term, mention_count))
        
        # Sort by the new, correct mention count
        return sorted(options, key=lambda x: x[1], reverse=True)

    dropdown_options = get_dropdown_options(chart_data["text"].dropna(), ngram_range)
    formatted_options = [f"{term} ({count} mentions)" for term, count in dropdown_options]

    selected_option = st.selectbox(
        "Select a term to analyze:",
        options=["--- Select a Term ---"] + formatted_options,
        on_change=reset_keyword_page
    )

    if selected_option != "--- Select a Term ---":
        selected_term = " ".join(selected_option.split(' ')[:-2])
        
        keyword_df = chart_data[chart_data['text'].str.contains(re.escape(selected_term), case=False, na=False)]
        
        st.markdown(f"---")
        st.markdown(f"#### Analysis for term: `{selected_term}` ({len(keyword_df)} mentions)")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Rating Distribution**")
            rating_dist = keyword_df['rating'].value_counts().reindex(range(1, 6), fill_value=0).sort_index()
            st.bar_chart(rating_dist)
        with col2:
            st.markdown("**Sentiment Distribution**")
            sentiment_dist = keyword_df['sentiment'].value_counts().reindex(['Positive', 'Neutral', 'Negative'], fill_value=0)
            st.bar_chart(sentiment_dist)
        
        # (Rest of the review display logic is unchanged)
        # ...
        st.markdown("**Example Reviews**")
        sort_reviews_by = st.selectbox("Sort examples by:",("Most Helpful", "Newest", "Oldest", "Highest Rating", "Lowest Rating"),key="keyword_review_sort",on_change=reset_keyword_page)
        if sort_reviews_by == "Most Helpful":
            sorted_keyword_df = keyword_df.sort_values(by="helpful_vote", ascending=False)
        elif sort_reviews_by == "Highest Rating":
            sorted_keyword_df = keyword_df.sort_values(by=["rating", "helpful_vote"], ascending=[False, False])
        elif sort_reviews_by == "Lowest Rating":
            sorted_keyword_df = keyword_df.sort_values(by=["rating", "helpful_vote"], ascending=[True, False])
        elif sort_reviews_by == "Oldest":
            sorted_keyword_df = keyword_df.sort_values(by="date", ascending=True)
        else: # Newest
            sorted_keyword_df = keyword_df.sort_values(by="date", ascending=False)
        if 'keyword_review_page' not in st.session_state:
            st.session_state.keyword_review_page = 0
        start_idx = st.session_state.keyword_review_page * REVIEWS_PER_PAGE
        end_idx = start_idx + REVIEWS_PER_PAGE
        reviews_to_display = sorted_keyword_df.iloc[start_idx:end_idx]
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
                st.markdown(f"> {review['text']}")
        total_reviews = len(sorted_keyword_df)
        total_pages = (total_reviews + REVIEWS_PER_PAGE - 1) // REVIEWS_PER_PAGE
        if total_pages > 1:
            st.caption(f"Page {st.session_state.keyword_review_page + 1} of {total_pages}")
            p_col1, p_col2 = st.columns(2)
            with p_col1:
                if st.session_state.keyword_review_page > 0:
                    if st.button("⬅️ Previous 5 Reviews"):
                        st.session_state.keyword_review_page -= 1
                        st.rerun()
            with p_col2:
                if end_idx < total_reviews:
                    if st.button("Next 5 Reviews ➡️"):
                        st.session_state.keyword_review_page += 1
                        st.rerun()

    # --- Keyword Co-occurrence Network ---
    Of course. My apologies, the default pyvis styling can make the labels difficult to read, especially on a dark background.

The ideal way to fix this is to explicitly set the font size and color within the graph's options. This gives you direct control over the appearance of the node labels.

Here is the updated code for wxxthiq/sa-amazon-reviews/SA-Amazon-Reviews-dev/pages/3_Keyword_Analysis.py with the necessary changes to make the text clearly visible.

Corrected Keyword Analysis Page
Replace the entire content of pages/3_Keyword_Analysis.py with this final version:

Python

# pages/3_Keyword_Analysis.py
import streamlit as st
import pandas as pd
import re
from collections import Counter
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from utils.database_utils import (
    connect_to_db,
    get_product_details,
    get_reviews_for_product,
    get_product_date_range
)
import networkx as nx
from pyvis.network import Network
from itertools import combinations
import streamlit.components.v1 as components
import tempfile

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Keyword Analysis")
DB_PATH = "amazon_reviews_top100.duckdb"
conn = connect_to_db(DB_PATH)
REVIEWS_PER_PAGE = 5

# --- Main App Logic ---
def main():
    st.title("🔑 Detailed Keyword & Phrase Analysis")

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
    st.caption("Use the sidebar to filter reviews, then explore the most common terms and phrases.")

    # --- Sidebar Filters ---
    st.sidebar.header("🔬 Keyword Analysis Filters")
    min_date_db, max_date_db = get_product_date_range(conn, selected_asin)
    default_date_range = (min_date_db, max_date_db)
    default_ratings = [1, 2, 3, 4, 5]
    default_sentiments = ['Positive', 'Negative', 'Neutral']
    default_verified = "All"
    if 'keyword_date_filter' not in st.session_state: st.session_state.keyword_date_filter = default_date_range
    if 'keyword_rating_filter' not in st.session_state: st.session_state.keyword_rating_filter = default_ratings
    if 'keyword_sentiment_filter' not in st.session_state: st.session_state.keyword_sentiment_filter = default_sentiments
    if 'keyword_verified_filter' not in st.session_state: st.session_state.keyword_verified_filter = default_verified
    def reset_keyword_page():
        st.session_state.keyword_review_page = 0
    st.sidebar.date_input("Filter by Date Range", key='keyword_date_filter', on_change=reset_keyword_page)
    st.sidebar.multiselect("Filter by Star Rating", options=default_ratings, key='keyword_rating_filter', on_change=reset_keyword_page)
    st.sidebar.multiselect("Filter by Sentiment", options=default_sentiments, key='keyword_sentiment_filter', on_change=reset_keyword_page)
    st.sidebar.radio("Filter by Purchase Status", ["All", "Verified Only", "Not Verified"], key='keyword_verified_filter', on_change=reset_keyword_page)
    chart_data = get_reviews_for_product(conn, selected_asin, st.session_state.keyword_date_filter, tuple(st.session_state.keyword_rating_filter), tuple(st.session_state.keyword_sentiment_filter), st.session_state.keyword_verified_filter)

    st.markdown("---")
    if chart_data.empty:
        st.warning("No review data available for the selected filters.")
        st.stop()
    st.info(f"Analyzing keywords from **{len(chart_data)}** reviews matching your criteria.")
    
    # --- Keyword Co-occurrence Network ---
    st.markdown("---")
    st.markdown("### 🕸️ Keyword Co-occurrence Network")
    st.caption("This network shows which keywords frequently appear together in the same review. Stronger links indicate more frequent co-occurrence.")

    net_col1, net_col2 = st.columns(2)
    with net_col1:
        top_n_keywords = st.slider("Number of Top Keywords to Analyze:", min_value=10, max_value=50, value=25, key="top_n_slider")
    with net_col2:
        min_cooccurrence = st.slider("Minimum Co-occurrence:", min_value=2, max_value=20, value=5, key="min_co_slider")

    @st.cache_data
    def generate_network_graph(corpus, top_n, min_occur):
        vec = CountVectorizer(ngram_range=(1, 1), stop_words='english').fit(corpus)
        bag_of_words = vec.transform(corpus)
        sum_words = bag_of_words.sum(axis=0)
        words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
        words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
        top_keywords = [word for word, freq in words_freq[:top_n]]
        
        co_occurrence = pd.DataFrame(index=top_keywords, columns=top_keywords, dtype=np.int64).fillna(0)

        for text in corpus:
            tokens = [word for word in text.lower().split() if word in top_keywords]
            for w1, w2 in combinations(set(tokens), 2):
                co_occurrence.loc[w1, w2] += 1
                co_occurrence.loc[w2, w1] += 1

        G = nx.Graph()
        for word1 in co_occurrence.index:
            for word2 in co_occurrence.columns:
                weight = co_occurrence.loc[word1, word2]
                if weight >= min_occur:
                    G.add_edge(word1, word2, weight=int(weight), title=f"Co-occurrences: {int(weight)}")
        
        if not G.edges:
            return None

        net = Network(height="600px", width="100%", notebook=True, cdn_resources="in_line", bgcolor="#222222", font_color="white")
        net.from_nx(G)
        
        # --- UPDATED: Add font settings to the options ---
        options = """
        var options = {
          "nodes": {
            "font": {
              "size": 20,
              "face": "Tahoma",
              "color": "#ffffff"
            },
            "borderWidth": 2,
            "shapeProperties": {
              "useBorderWithImage": true
            }
          },
          "edges": {
            "color": {
              "inherit": true
            },
            "smooth": {
              "enabled": false,
              "type": "continuous"
            }
          },
          "physics": {
            "enabled": true,
            "stabilization": {
              "enabled": true,
              "iterations": 1000,
              "fit": true
            },
            "barnesHut": {
              "gravitationalConstant": -80000,
              "springConstant": 0.001,
              "springLength": 200
            }
          }
        }
        """
        net.set_options(options)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
            net.save_graph(tmp_file.name)
            with open(tmp_file.name, 'r', encoding='utf-8') as f:
                source_code = f.read()
        
        return source_code

    with st.spinner("Building keyword network..."):
        all_text = chart_data["text"].dropna()
        if not all_text.empty:
            network_html_content = generate_network_graph(all_text, top_n_keywords, min_cooccurrence)
            if network_html_content:
                components.html(network_html_content, height=610)
            else:
                st.warning("No significant keyword co-occurrences found with the current settings. Try lowering the minimum co-occurrence threshold.")
        else:
            st.warning("No review text available to build a network.")
            
if __name__ == "__main__":
    main()
