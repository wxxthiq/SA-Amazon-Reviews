# database_builder.py (DuckDB Version - Top 100 Products)
# This script creates a DuckDB database with the top 100 most popular products
# from each category to enable rapid experimentation and development.

import pandas as pd
import duckdb
import json
import os
import time
from textblob import TextBlob

# --- Configuration ---
KAGGLE_INPUT_DIR = '/kaggle/input/mcauley-jsonl'
OUTPUT_FOLDER = '/kaggle/working/'
# Use a .duckdb extension for the new database file
OUTPUT_DB_PATH = os.path.join(OUTPUT_FOLDER, 'amazon_reviews_top100.duckdb')

# Define the categories to process
CATEGORIES_TO_PROCESS = {
    'Amazon Fashion': ('Amazon_Fashion.jsonl', 'meta_Amazon_Fashion.jsonl'),
    'All Beauty': ('All_Beauty.jsonl', 'meta_All_Beauty.jsonl'),
    'Appliances': ('Appliances.jsonl', 'meta_Appliances.jsonl')
}
# The number of top products to select from each category
TOP_N_PRODUCTS = 100

# --- Helper Functions ---
def parse_jsonl_to_df(path):
    """Reads a JSON-lines (.jsonl) file into a pandas DataFrame."""
    print(f"  Reading data from: {os.path.basename(path)}")
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
    """Safely extracts all available image URLs and joins them into a comma-separated string."""
    if not isinstance(image_data, list) or not image_data: return None
    urls = [img.get('hi_res') or img.get('large') for img in image_data if isinstance(img, dict) and (img.get('hi_res') or img.get('large'))]
    return ",".join(urls) if urls else None

# --- Main Build Process ---
def main():
    start_time = time.time()
    print(f"--- Starting Database Build: {OUTPUT_DB_PATH} ---")

    # Use duckdb.connect() which creates or opens the database file
    conn = duckdb.connect(database=OUTPUT_DB_PATH, read_only=False)

    all_processed_reviews = []

    # --- STAGE 1: Identify Top Products and Process Their Reviews ---
    for category_name, (review_file, meta_file) in CATEGORIES_TO_PROCESS.items():
        phase_start_time = time.time()
        print(f"\n--- Processing Category: {category_name} ---")

        review_file_path = os.path.join(KAGGLE_INPUT_DIR, review_file, review_file)
        meta_file_path = os.path.join(KAGGLE_INPUT_DIR, meta_file, meta_file)

        # 1. Load all reviews for the category to find the most popular products
        print(f"  - Loading all reviews to identify top {TOP_N_PRODUCTS} products...")
        reviews_df = parse_jsonl_to_df(review_file_path)
        if reviews_df is None or reviews_df.empty:
            continue

        # Find the top N products by review count
        top_products = reviews_df['parent_asin'].value_counts().nlargest(TOP_N_PRODUCTS).index.tolist()
        print(f"  - Identified top {len(top_products)} products.")

        # Filter the reviews to only include those for the top products
        reviews_df = reviews_df[reviews_df['parent_asin'].isin(top_products)]

        # 2. Load and prepare metadata
        meta_df = parse_jsonl_to_df(meta_file_path)
        if meta_df is None: continue

        # Rename columns for clarity and select only the necessary ones
        meta_df = meta_df.rename(columns={'title': 'product_title', 'images': 'image_list'})
        meta_df['image_urls'] = meta_df['image_list'].apply(extract_and_join_image_urls)
        meta_df_filtered = meta_df[['parent_asin', 'product_title', 'image_urls', 'store']].dropna(subset=['parent_asin'])

        # 3. Merge reviews with metadata
        print("  - Merging reviews with product metadata...")
        merged_df = pd.merge(reviews_df, meta_df_filtered, on='parent_asin', how='left')

        # 4. Enrich the data (Sentiment, IDs, etc.)
        print("  - Enriching data with sentiment, IDs, and cleaning...")
        merged_df['category'] = category_name
        merged_df['rating'] = pd.to_numeric(merged_df['rating'], errors='coerce')
        # Drop rows with essential missing data
        merged_df.dropna(subset=['rating', 'text', 'parent_asin'], inplace=True)

        # Create a stable, unique review_id
        merged_df.reset_index(drop=True, inplace=True)
        merged_df['review_id'] = merged_df['parent_asin'] + '-' + merged_df.index.astype(str)

        # Perform sentiment analysis
        sentiments = merged_df['text'].astype(str).apply(lambda text: TextBlob(text).sentiment)
        merged_df['sentiment'] = sentiments.apply(lambda s: 'Positive' if s.polarity > 0.1 else ('Negative' if s.polarity < -0.1 else 'Neutral'))
        merged_df['text_polarity'] = sentiments.apply(lambda s: s.polarity)
        merged_df['date'] = pd.to_datetime(merged_df['timestamp'], unit='ms', errors='coerce').dt.date
        merged_df['helpful_vote'] = pd.to_numeric(merged_df['helpful_vote'], errors='coerce').fillna(0).astype(int)
        
        # Rename the 'title' from the review file to 'review_title'
        merged_df.rename(columns={'title': 'review_title'}, inplace=True)

        # Select the final columns to keep
        final_columns = [
            'review_id', 'parent_asin', 'product_title', 'category', 'store', 'image_urls',
            'rating', 'review_title', 'text', 'sentiment', 'text_polarity', 'date',
            'helpful_vote', 'verified_purchase'
        ]
        
        # Ensure all columns exist, fill missing with None
        for col in final_columns:
            if col not in merged_df:
                merged_df[col] = None

        all_processed_reviews.append(merged_df[final_columns])
        print(f"✅ Finished '{category_name}' in {time.time() - phase_start_time:.2f} seconds.")

    if not all_processed_reviews:
        print("❌ No reviews were processed. Exiting.")
        return

    # Combine all processed data into a single DataFrame
    final_df = pd.concat(all_processed_reviews, ignore_index=True)

    # --- STAGE 2: Create Final Tables in DuckDB ---
    print("\n--- STAGE 2: Building Final Tables in DuckDB ---")

    # Register the final DataFrame as a "virtual table" in DuckDB
    conn.register('raw_reviews_view', final_df)

    # Products Table
    print("-> Building 'products' table...")
    conn.execute("""
        CREATE OR REPLACE TABLE products AS
        SELECT
            parent_asin,
            FIRST(product_title) as product_title,
            FIRST(category) as category,
            FIRST(store) as store,
            FIRST(image_urls) as image_urls,
            AVG(rating) as average_rating,
            COUNT(review_id) as review_count
        FROM raw_reviews_view
        GROUP BY parent_asin;
    """)

    # Reviews Table
    print("-> Building 'reviews' table...")
    conn.execute("""
        CREATE OR REPLACE TABLE reviews AS
        SELECT
            review_id,
            parent_asin,
            rating,
            review_title,
            text,
            sentiment,
            text_polarity,
            date,
            helpful_vote,
            verified_purchase
        FROM raw_reviews_view;
    """)

    # We don't need a separate discrepancy table anymore,
    # as the 'reviews' table now contains all necessary columns.

    conn.unregister('raw_reviews_view') # Clean up the virtual table
    print("✅ All tables created successfully.")

    conn.close()
    end_time = time.time()
    print(f"\n✅✅ Database build complete in {end_time - start_time:.2f} seconds.")
    print(f"Database file created at: {OUTPUT_DB_PATH}")

if __name__ == '__main__':
    # NOTE: This script assumes the Kaggle dataset is in a directory named 'mcauley-jsonl'
    # in the same root directory as the script.
    # Adjust the KAGGLE_INPUT_DIR path if your structure is different.
    if not os.path.exists(KAGGLE_INPUT_DIR):
        print(f"Error: Input directory '{KAGGLE_INPUT_DIR}' not found.")
        print("Please ensure the dataset is downloaded and placed in the correct directory.")
    else:
        main()
