# utils/database_utils.py
import streamlit as st
import duckdb
import pandas as pd
import os
# We will import kaggle and json inside the function that needs them
import logging
from datetime import datetime

# --- Configure logging ---
logging.basicConfig(level=logging.INFO)
VERSION_FILE_PATH = ".db_version"

@st.cache_resource
def connect_to_db(path):
    """Connects to the DuckDB database."""
    try:
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
        # --- LAZY IMPORT ---
        # Import kaggle and json here so it only happens when this function is called.
        import kaggle
        import json
        
        kaggle_dir = os.path.expanduser("~/.kaggle")
        os.makedirs(kaggle_dir, exist_ok=True)
        kaggle_json_path = os.path.join(kaggle_dir, "kaggle.json")

        if "kaggle" not in st.secrets or "username" not in st.secrets["kaggle"] or "key" not in st.secrets["kaggle"]:
            st.error('FATAL: Make sure your .streamlit/secrets.toml contains a [kaggle] section with "username" and "key".')
            st.stop()

        credentials = {"username": st.secrets["kaggle"]["username"], "key": st.secrets["kaggle"]["key"]}

        with open(kaggle_json_path, "w") as f:
            json.dump(credentials, f)
        os.chmod(kaggle_json_path, 0o600)

        with st.spinner(f"Downloading dataset '{dataset_slug}' from Kaggle..."):
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

def get_filtered_products(_conn, category, search_term, sort_by, limit, offset):
    """Fetches a paginated and filtered list of products using positional placeholders."""
    params = [category]

    where_clauses = ["category = ?"]
    if search_term:
        where_clauses.append("product_title ILIKE ?")
        params.append(f"%{search_term}%")

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
def get_reviews_for_product(_conn, asin, date_range, rating_filter, sentiment_filter):
    """Fetches and filters all reviews for a single product using positional placeholders."""
    query = "SELECT * FROM reviews WHERE parent_asin = ?"
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

    return _conn.execute(query, params).fetchdf()

@st.cache_data
def get_product_date_range(_conn, asin):
    """Gets the min and max review dates for a product."""
    res = _conn.execute("SELECT MIN(date), MAX(date) FROM reviews WHERE parent_asin = ?", [asin]).fetchone()
    min_date = res[0] if res[0] else datetime.now().date()
    max_date = res[1] if res[1] else datetime.now().date()
    return min_date, max_date

@st.cache_data
def get_single_review_text(_conn, review_id):
    """Fetches the full text of a single review by its unique ID."""
    try:
        result = _conn.execute("SELECT text FROM reviews WHERE review_id = ?", [review_id]).fetchone()
        return result[0] if result else "Review text not found."
    except Exception:
        return "Could not retrieve review text."
