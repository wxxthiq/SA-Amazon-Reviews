# utils/database_utils.py
import streamlit as st
import duckdb
import pandas as pd
import os
import kaggle
import json
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
        # Using Streamlit secrets for Kaggle credentials
        kaggle_dir = os.path.expanduser("~/.kaggle")
        os.makedirs(kaggle_dir, exist_ok=True)
        kaggle_json_path = os.path.join(kaggle_dir, "kaggle.json")
        
        credentials = {"username": st.secrets["kaggle"]["username"], "key": st.secrets["kaggle"]["key"]}
        with open(kaggle_json_path, "w") as f:
            json.dump(credentials, f)
        os.chmod(kaggle_json_path, 0o600)

        with st.spinner(f"Downloading dataset '{dataset_slug}' from Kaggle..."):
            kaggle.api.dataset_download_files(dataset=dataset_slug, path='.', unzip=True)
        
        # After download, write the new version number
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

def get_filtered_products(_conn, category, search_term, sort_by, limit, offset):
    """Fetches a paginated and filtered list of products."""
    params = {'category': category}
    
    where_clauses = ["category = :category"]
    if search_term:
        where_clauses.append("product_title ILIKE :search_term")
        params['search_term'] = f"%{search_term}%"

    where_sql = " WHERE " + " AND ".join(where_clauses)
    
    base_query = f"FROM products {where_sql}"
    
    count_query = f"SELECT COUNT(*) {base_query}"
    total_count = _conn.execute(count_query, params).fetchone()[0]

    order_by_sql = {
        "Popularity (Most Reviews)": "review_count DESC",
        "Highest Rating": "average_rating DESC",
        "Lowest Rating": "average_rating ASC"
    }.get(sort_by, "review_count DESC")

    query = f"""
        SELECT * {base_query}
        ORDER BY {order_by_sql}
        LIMIT :limit OFFSET :offset
    """
    params['limit'] = limit
    params['offset'] = offset
    
    df = _conn.execute(query, params).fetchdf()
    return df, total_count

@st.cache_data
def get_product_details(_conn, asin):
    """Fetches all details for a single product."""
    return _conn.execute("SELECT * FROM products WHERE parent_asin = ?", [asin]).fetchdf()

@st.cache_data
def get_reviews_for_product(_conn, asin, date_range, rating_filter, sentiment_filter):
    """Fetches and filters all reviews for a single product for analysis."""
    query = "SELECT * FROM reviews WHERE parent_asin = :asin"
    params = {'asin': asin}

    if date_range and len(date_range) == 2:
        start_date = date_range[0]
        end_date = date_range[1]
        query += " AND date BETWEEN :start_date AND :end_date"
        params['start_date'] = start_date
        params['end_date'] = end_date
    
    if rating_filter:
        # DuckDB requires a tuple for the IN clause
        query += " AND rating IN :rating_filter"
        params['rating_filter'] = tuple(rating_filter)
        
    if sentiment_filter:
        query += " AND sentiment IN :sentiment_filter"
        params['sentiment_filter'] = tuple(sentiment_filter)

    return _conn.execute(query, params).fetchdf()

@st.cache_data
def get_product_date_range(_conn, asin):
    """Gets the min and max review dates for a product."""
    res = _conn.execute("SELECT MIN(date), MAX(date) FROM reviews WHERE parent_asin = ?", [asin]).fetchone()
    min_date = res[0] if res[0] else datetime.now().date()
    max_date = res[1] if res[1] else datetime.now().date()
    return min_date, max_date
