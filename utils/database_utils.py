# utils/database_utils.py
import streamlit as st
import duckdb
import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime

# --- Configure logging ---
logging.basicConfig(level=logging.INFO)
VERSION_FILE_PATH = ".db_version"

@st.cache_resource
def connect_to_db(path):
    """Connects to the DuckDB database."""
    try:
        # Connect in read-only mode as the app shouldn't modify the DB
        return duckdb.connect(database=path, read_only=True)
    except Exception as e:
        st.error(f"FATAL: Could not connect to database at '{path}'. Error: {e}")
        st.stop()

def a_download_data_with_versioning(dataset_slug, db_path, expected_version):
    """Downloads data from Kaggle if the local version is outdated."""
    current_version = 0
    if os.path.exists(VERSION_FILE_PATH):
        with open(VERSION_FILE_PATH, "r") as f:
            try:
                current_version = int(f.read().strip())
            except (ValueError, TypeError):
                current_version = 0

    if current_version == expected_version and os.path.exists(db_path):
        logging.info("Database is up to date.")
        return

    st.info(f"Database v{current_version} is outdated (expected v{expected_version}). Downloading fresh data...")
    if os.path.exists(db_path): os.remove(db_path)
    if os.path.exists(VERSION_FILE_PATH): os.remove(VERSION_FILE_PATH)

    try:
        # --- ROBUST AUTHENTICATION FIX ---
        # Set credentials as environment variables. Kaggle's library will automatically find them.
        # This avoids all issues with creating the kaggle.json file in a cloud environment.
        if "kaggle" not in st.secrets or "username" not in st.secrets["kaggle"] or "key" not in st.secrets["kaggle"]:
            st.error('FATAL: Make sure your .streamlit/secrets.toml contains a [kaggle] section with "username" and "key".')
            st.stop()

        os.environ['KAGGLE_USERNAME'] = st.secrets["kaggle"]["username"]
        os.environ['KAGGLE_KEY'] = st.secrets["kaggle"]["key"]

        # Now that environment variables are set, we can safely import kaggle.
        import kaggle

        with st.spinner(f"Downloading dataset '{dataset_slug}' from Kaggle..."):
            # The Kaggle API will now use the environment variables for authentication.
            kaggle.api.dataset_download_files(dataset=dataset_slug, path='.', unzip=True)

        with open(VERSION_FILE_PATH, "w") as f:
            f.write(str(expected_version))
        st.success("Database download complete! Rerunning app...")
        st.rerun()

    except Exception as e:
        st.error(f"FATAL: Error during Kaggle download: {e}")
        logging.error(f"Kaggle API error: {e}")
        st.stop()

@st.cache_data
def get_all_categories(_conn):
    """Fetches a list of all unique product categories."""
    df = _conn.execute("SELECT DISTINCT category FROM products ORDER BY category").fetchdf()
    categories = df['category'].dropna().tolist()
    categories.insert(0, "--- Select a Category ---")
    return categories

@st.cache_data
def get_filtered_products(_conn, category, search_term, sort_by, rating_range, review_count_range, limit, offset):
    """Fetches a paginated and filtered list of products using positional placeholders."""
    params = [category]

    where_clauses = ["category = ?"]
    if search_term:
        where_clauses.append("product_title ILIKE ?")
        params.append(f"%{search_term}%")

    # --- NEW: Add rating and review count filters ---
    if rating_range:
        where_clauses.append("average_rating BETWEEN ? AND ?")
        params.extend([rating_range[0], rating_range[1]])

    if review_count_range:
        where_clauses.append("review_count BETWEEN ? AND ?")
        params.extend([review_count_range[0], review_count_range[1]])


    where_sql = " WHERE " + " AND ".join(where_clauses)

    count_query = f"SELECT COUNT(*) FROM products {where_sql}"
    total_count = _conn.execute(count_query, params).fetchone()[0]

    order_by_sql = {
        "Popularity (Most Reviews)": "review_count DESC",
        "Highest Rating": "average_rating DESC",
        "Lowest Rating": "average_rating ASC"
    }.get(sort_by, "review_count DESC")

    query = f"""
        SELECT * FROM products {where_sql}
        ORDER BY {order_by_sql}
        LIMIT ? OFFSET ?
    """
    params.extend([limit, offset])

    df = _conn.execute(query, params).fetchdf()
    return df, total_count

@st.cache_data
def get_product_details(_conn, asin):
    """Fetches all details for a single product."""
    return _conn.execute("SELECT * FROM products WHERE parent_asin = ?", [asin]).fetchdf()

@st.cache_data
def get_reviews_for_product(_conn, asin, date_range, rating_filter, sentiment_filter, verified_filter):
    """
    Fetches and filters all reviews, now with verified purchase filter.
    """
    query = "SELECT review_id, parent_asin, rating, sentiment, text_polarity, text, date, verified_purchase, review_title FROM reviews WHERE parent_asin = ?"
    params = [asin]

    if date_range and len(date_range) == 2:
        start_date, end_date = date_range
        query += " AND date BETWEEN ? AND ?"
        params.extend([start_date, end_date])
    
    if rating_filter:
        placeholders = ', '.join(['?'] * len(rating_filter))
        query += f" AND rating IN ({placeholders})"
        params.extend(rating_filter)
        
    if sentiment_filter:
        placeholders = ', '.join(['?'] * len(sentiment_filter))
        query += f" AND sentiment IN ({placeholders})"
        params.extend(sentiment_filter)
        
    # ** NEW: Add verified purchase filter logic **
    if verified_filter == "Verified Only":
        query += " AND verified_purchase = TRUE"
    elif verified_filter == "Not Verified":
        query += " AND verified_purchase = FALSE"

    df = _conn.execute(query, params).fetchdf()

    if not df.empty:
        rng = np.random.default_rng(seed=42)
        df['rating_jittered'] = df['rating'] + rng.uniform(-0.1, 0.1, size=len(df))
        df['text_polarity_jittered'] = df['text_polarity'] + rng.uniform(-0.02, 0.02, size=len(df))

    return df
    
@st.cache_data
def get_product_date_range(_conn, asin):
    """Gets the min and max review dates for a product."""
    res = _conn.execute("SELECT MIN(date), MAX(date) FROM reviews WHERE parent_asin = ?", [asin]).fetchone()
    min_date = res[0] if res[0] else datetime.now().date()
    max_date = res[1] if res[1] else datetime.now().date()
    return min_date, max_date

@st.cache_data
def get_single_review_details(_conn, review_id):
    """Fetches all details for a single review by its ID."""
    try:
        query = "SELECT review_title, text, date, verified_purchase, helpful_vote, sentiment, text_polarity FROM reviews WHERE review_id = ?"
        details_df = _conn.execute(query, [review_id]).fetchdf()
        if not details_df.empty:
            return details_df.iloc[0]
        else:
            return None
    except Exception:
        return None

@st.cache_data
def get_single_review_details(_conn, review_id):
    """Fetches all details for a single review by its ID."""
    try:
        # Fetch title, text, and date for the selected review
        query = "SELECT review_title, text, date FROM reviews WHERE review_id = ?"
        details_df = _conn.execute(query, [review_id]).fetchdf()
        if not details_df.empty:
            return details_df.iloc[0]
        else:
            return None
    except Exception:
        return None

def get_paginated_reviews(_conn, asin, date_range, rating_filter, sentiment_filter, verified_filter, search_term, sort_by, limit, offset):
    """
    Fetches paginated reviews with updated sorting, search, and returns the full dataset for export.
    """
    query = "FROM reviews WHERE parent_asin = ?"
    params = [asin]

    # --- Add filters to the query ---
    if date_range and len(date_range) == 2:
        start_date, end_date = date_range
        query += " AND date BETWEEN ? AND ?"
        params.extend([start_date, end_date])
    
    if rating_filter:
        placeholders = ', '.join(['?'] * len(rating_filter))
        query += f" AND rating IN ({placeholders})"
        params.extend(rating_filter)
        
    if sentiment_filter:
        placeholders = ', '.join(['?'] * len(sentiment_filter))
        query += f" AND sentiment IN ({placeholders})"
        params.extend(sentiment_filter)
        
    if verified_filter == "Verified Only":
        query += " AND verified_purchase = TRUE"
    elif verified_filter == "Not Verified":
        query += " AND verified_purchase = FALSE"
        
    # --- NEW: Add keyword search functionality ---
    if search_term:
        query += " AND text ILIKE ?"
        params.append(f"%{search_term}%")

    # --- Fetch the full, filtered dataset for the export button ---
    full_filtered_query = f"SELECT * {query}"
    # We apply a reasonable limit for export to prevent memory issues
    all_filtered_df = _conn.execute(full_filtered_query + " LIMIT 5000", params).fetchdf() 
    
    total_reviews = len(all_filtered_df)

    # --- Sorting logic ---
    sort_logic = {
        "Newest First": "verified_purchase DESC, date DESC",
        "Oldest First": "verified_purchase DESC, date ASC",
        "Highest Rating": "verified_purchase DESC, rating DESC, helpful_vote DESC",
        "Lowest Rating": "verified_purchase DESC, rating ASC, helpful_vote DESC",
        "Most Helpful": "helpful_vote DESC, rating DESC"
    }
    order_by_sql = sort_logic.get(sort_by, "verified_purchase DESC, date DESC")

    # --- Final paginated query ---
    final_query = f"SELECT * {query} ORDER BY {order_by_sql} LIMIT ? OFFSET ?"
    params.extend([limit, offset])
    paginated_reviews_df = _conn.execute(final_query, params).fetchdf()

    # --- RETURN a tuple with all the necessary data ---
    return paginated_reviews_df, total_reviews, all_filtered_df
