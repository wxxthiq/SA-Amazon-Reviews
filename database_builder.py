# database_builder.py (Final, Interactive, Future-Proof Version)
# This script creates all tables with the necessary columns and IDs to support a
# fast, stable, and highly interactive Streamlit dashboard with rich filtering.

import pandas as pd
import sqlite3
from textblob import TextBlob
import spacy
import time
import os
import json

# --- Configuration ---
KAGGLE_INPUT_DIR = 'mcauley-jsonl'
OUTPUT_FOLDER = '/kaggle/working/'
OUTPUT_DB_PATH = os.path.join(OUTPUT_FOLDER, 'amazon_reviews_v5.db')

CATEGORIES_TO_PROCESS = {
    'Amazon Fashion': ('Amazon_Fashion.jsonl', 'meta_Amazon_Fashion.jsonl'),
    'All Beauty': ('All_Beauty.jsonl', 'meta_All_Beauty.jsonl'),
    'Appliances': ('Appliances.jsonl', 'meta_Appliances.jsonl')
}
CHUNK_SIZE = 150000

# --- Helper Functions ---
def parse_jsonl_to_df(path):
    """Reads a plain text JSON-lines (.jsonl) file."""
    print(f"  Reading metadata from: {path}")
    data = []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
        return pd.DataFrame.from_records(data)
    except Exception as e:
        print(f"    ❌ ERROR reading file {path}: {e}")
        return None

def extract_and_join_image_urls(image_data):
    """Safely extracts all available URLs and joins them into a comma-separated string."""
    if not isinstance(image_data, list) or not image_data: return None
    urls = [img.get('hi_res') or img.get('large') for img in image_data if isinstance(img, dict) and (img.get('hi_res') or img.get('large'))]
    return ",".join(urls) if urls else None

# --- Modular Build Functions ---

def stage_raw_data(conn, full_input_path):
    """
    Reads all source JSONL files, enriches them with a unique review_id and all
    necessary fields for filtering, and stages them into a single 'raw_reviews' table.
    """
    print("\n--- STAGE 1: Staging Raw Data with Unique IDs and All Filterable Fields ---")
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='raw_reviews'")
    if cursor.fetchone():
        print("✅ 'raw_reviews' table already exists. Skipping.")
        return

    is_first_write = True
    for category_name, (review_file, meta_file) in CATEGORIES_TO_PROCESS.items():
        phase_start_time = time.time()
        print(f"\n--- Processing Category: {category_name} ---")
        
        review_file_path = os.path.join(full_input_path, review_file, review_file)
        meta_file_path = os.path.join(full_input_path, meta_file, meta_file)

        meta_df = parse_jsonl_to_df(meta_file_path)
        if meta_df is None: continue

        meta_df = meta_df.rename(columns={'title': 'product_title', 'images': 'image_list'})
        meta_df['image_urls'] = meta_df['image_list'].apply(extract_and_join_image_urls)
        meta_df = meta_df[['parent_asin', 'product_title', 'image_urls']].dropna(subset=['parent_asin'])

        chunk_num = 1
        for reviews_chunk_df in pd.read_json(review_file_path, lines=True, chunksize=CHUNK_SIZE):
            print(f"  - Processing chunk {chunk_num}...")
            
            merged_chunk_df = pd.merge(reviews_chunk_df, meta_df, on='parent_asin', how='left')
            merged_chunk_df['category'] = category_name
            merged_chunk_df['rating'] = pd.to_numeric(merged_chunk_df['rating'], errors='coerce')
            merged_chunk_df.dropna(subset=['rating', 'text', 'parent_asin'], inplace=True)
            
            # --- KEY MODIFICATION: Create a unique review_id ---
            merged_chunk_df.reset_index(inplace=True)
            merged_chunk_df['review_id'] = merged_chunk_df['parent_asin'] + '-' + merged_chunk_df['index'].astype(str) + '-' + str(chunk_num)
            
            sentiments = merged_chunk_df['text'].astype(str).apply(lambda text: TextBlob(text).sentiment)
            merged_chunk_df['sentiment'] = sentiments.apply(lambda s: 'Positive' if s.polarity > 0.1 else ('Negative' if s.polarity < -0.1 else 'Neutral'))
            merged_chunk_df['text_polarity'] = sentiments.apply(lambda s: s.polarity)
            merged_chunk_df['date'] = pd.to_datetime(merged_chunk_df['timestamp'], unit='s', errors='coerce').dt.strftime('%Y-%m-%d')

            final_columns = ['review_id', 'parent_asin', 'product_title', 'category', 'image_urls', 'rating', 'text', 'sentiment', 'text_polarity', 'date']
            
            write_mode = 'replace' if is_first_write and chunk_num == 1 else 'append'
            merged_chunk_df[final_columns].to_sql('raw_reviews', conn, if_exists=write_mode, index=False)
            
            is_first_write = False
            chunk_num += 1
        print(f"✅ Finished staging '{category_name}' in {time.time() - phase_start_time:.2f} seconds.")
    print("✅ Raw data staging complete.")


def create_final_tables(conn):
    """
    Creates all the final, optimized tables from the 'raw_reviews' staging table.
    """
    print("\n--- STAGE 2: Creating Final, Optimized Tables ---")
    cursor = conn.cursor()

    # Products Table
    print("-> Building 'products' table...")
    cursor.execute("DROP TABLE IF EXISTS products")
    cursor.execute("""
        CREATE TABLE products AS SELECT parent_asin,
            FIRST_VALUE(product_title) OVER (PARTITION BY parent_asin ORDER BY rowid) as product_title,
            FIRST_VALUE(image_urls) OVER (PARTITION BY parent_asin ORDER BY rowid) as image_urls,
            FIRST_VALUE(category) OVER (PARTITION BY parent_asin ORDER BY rowid) as category,
            AVG(rating) as average_rating, COUNT(rating) as review_count
        FROM raw_reviews GROUP BY parent_asin;
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_products_parent_asin ON products (parent_asin);")

    # Reviews Table (for drill-down and pagination)
    print("-> Building 'reviews' table...")
    cursor.execute("DROP TABLE IF EXISTS reviews")
    cursor.execute("""
        CREATE TABLE reviews AS
        SELECT review_id, parent_asin, rating, sentiment, text, date FROM raw_reviews;
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_reviews_review_id ON reviews (review_id);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_reviews_parent_asin ON reviews (parent_asin);")
    
    # Discrepancy Table (now with all data needed for filtering and drill-down)
    print("-> Building 'discrepancy_data' table...")
    cursor.execute("DROP TABLE IF EXISTS discrepancy_data")
    cursor.execute("""
        CREATE TABLE discrepancy_data AS
        SELECT review_id, parent_asin, rating, text_polarity, sentiment, date FROM raw_reviews;
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_discrepancy_parent_asin ON discrepancy_data (parent_asin);")
    
    # Rating Distribution Table
    print("-> Building 'rating_distribution' table...")
    cursor.execute("DROP TABLE IF EXISTS rating_distribution")
    cursor.execute("""
        CREATE TABLE rating_distribution AS SELECT parent_asin,
            COUNT(CASE WHEN rating = 1.0 THEN 1 END) as '1_star',
            COUNT(CASE WHEN rating = 2.0 THEN 1 END) as '2_star',
            COUNT(CASE WHEN rating = 3.0 THEN 1 END) as '3_star',
            COUNT(CASE WHEN rating = 4.0 THEN 1 END) as '4_star',
            COUNT(CASE WHEN rating = 5.0 THEN 1 END) as '5_star'
        FROM raw_reviews GROUP BY parent_asin;
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_rating_dist_parent_asin ON rating_distribution (parent_asin);")

    conn.commit()
    print("✅ Core tables created successfully.")

# --- Main Execution Controller ---
def main():
    start_time = time.time()
    conn = sqlite3.connect(OUTPUT_DB_PATH)
    
    # Run the staging process
    stage_raw_data(conn, '/kaggle/input/' + KAGGLE_INPUT_DIR)
    
    # Create all final tables from the staged data
    create_final_tables(conn)
    
    # NOTE: Aspect analysis is not included in this build for speed,
    # but can be added later as a new modular function.
    
    # Final cleanup
    print("\n--- Finalizing Database ---")
    conn.execute("DROP TABLE IF EXISTS raw_reviews;")
    conn.execute("VACUUM;")
    conn.commit()
    print("✅ Cleanup complete.")
    
    conn.close()
    end_time = time.time()
    print(f"\n✅ Database build complete in {end_time - start_time:.2f} seconds.")

if __name__ == '__main__':
    main()
