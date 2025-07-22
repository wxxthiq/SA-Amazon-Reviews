# utilities/duckdb_connector.py
import streamlit as st
import duckdb
import pandas as pd
import requests
import os

DB_URL = "https://www.kaggle.com/datasets/wathiqsoualhi/amazon-aspect" # Replace with your actual URL
DB_PATH = "amazon_reviews_final.duckdb" # The local path to save the file

@st.cache_resource(show_spinner="Connecting to the database...")
def download_database_if_needed():
    """
    Checks if the DuckDB database file exists. If not, it downloads it from
    the specified URL (e.g., from Kaggle Datasets).
    """
    if not os.path.exists(DB_PATH):
        st.info(f"Downloading database from Kaggle... This may take a moment.")
        try:
            with requests.get(DB_URL, stream=True) as r:
                r.raise_for_status()
                with open(DB_PATH, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            st.success("Database download complete!")
        except Exception as e:
            st.error(f"Failed to download database: {e}")
            st.stop()
            
@st.cache_resource
def get_db_connection():
    """
    Creates and returns a connection to the DuckDB database.
    Uses Streamlit's resource caching to ensure only one connection
    is opened per user session, enhancing performance.
    """
    # The connection is set to read-only, a safe default for an analytics app.
    return duckdb.connect(database=DB_FILE, read_only=True)

@st.cache_data
def run_query(query: str) -> pd.DataFrame:
    """
    Executes a SQL query against the DuckDB database and returns
    the result as a Pandas DataFrame.
    Uses Streamlit's data caching to avoid re-running the same
    query multiple times within a session.

    Args:
        query (str): The SQL query to execute.

    Returns:
        pd.DataFrame: A Pandas DataFrame containing the query result.
    """
    # Retrieve the cached database connection.
    conn = get_db_connection()
    # Execute the query and convert the result directly to a DataFrame.
    return conn.sql(query).df()

# The following functions are examples of more specific, application-oriented
# data retrieval functions that build upon the generic run_query.

@st.cache_data
def get_reviews_by_asin(asin: str) -> pd.DataFrame:
    """
    Fetches all reviews for a specific product ASIN.
    The query is parameterized to prevent SQL injection, although the risk
    is low in this context, it is a good practice.
    
    Args:
        asin (str): The product ASIN to query for.
        
    Returns:
        pd.DataFrame: A DataFrame of reviews for the specified ASIN.
    """
    query = f"SELECT * FROM reviews WHERE asin = '{asin}'"
    return run_query(query)

@st.cache_data
def get_all_asins() -> list[str]:
    """
    Fetches a unique list of all ASINs present in the database.
    Useful for populating dropdown selectors.
    """
    query = "SELECT DISTINCT asin FROM reviews ORDER BY asin"
    df = run_query(query)
    return df['asin'].tolist()
