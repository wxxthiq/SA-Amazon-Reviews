# database_builder.py (Final Modular & Updatable Version)
# This script allows re-running to add new features without rebuilding everything from scratch.
# Includes category-specific aspects and rating distribution pre-computation.

import pandas as pd
import sqlite3
from textblob import TextBlob
import spacy
import time
import os
import json

# --- Configuration ---
# This should match the directory where your JSONL files are located on your build machine (e.g., Kaggle)
KAGGLE_INPUT_DIR = 'mcauley-jsonl'
OUTPUT_FOLDER = '/kaggle/working/'
# I've named the output DB 'final_v2.db' to distinguish it from previous versions.
OUTPUT_DB_PATH = os.path.join(OUTPUT_FOLDER, 'amazon_reviews_v3.db')

CATEGORIES_TO_PROCESS = {
    'Amazon Fashion': ('Amazon_Fashion.jsonl', 'meta_Amazon_Fashion.jsonl'),
    'All Beauty': ('All_Beauty.jsonl', 'meta_All_Beauty.jsonl'),
    'Appliances': ('Appliances.jsonl', 'meta_Appliances.jsonl')
}
CHUNK_SIZE = 100000 # Reduced for memory safety during the build process

# --- PERFECTED ASPECT LISTS ---
# Aspects common to all product types
COMMON_ASPECTS = ['price', 'value', 'quality', 'packaging', 'shipping', 'delivery', 'service', 'return', 'customer service']

# Category-specific aspects
FASHION_ASPECTS = ['fit', 'size', 'color', 'fabric', 'style', 'comfort', 'stitching', 'zipper', 'material', 'durability', 'design', 'look']
BEAUTY_ASPECTS = ['scent', 'fragrance', 'texture', 'consistency', 'longevity', 'coverage', 'formula', 'ingredients', 'application', 'pigmentation', 'moisture']
APPLIANCES_ASPECTS = ['power', 'noise', 'performance', 'battery', 'durability', 'ease of use', 'features', 'design', 'size', 'installation', 'efficiency']

# Combine all aspects into a single list for analysis
ALL_ASPECTS = list(set(COMMON_ASPECTS + FASHION_ASPECTS + BEAUTY_ASPECTS + APPLIANCES_ASPECTS))


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
    Reads all source JSONL files, enriches them, and stages them into a single 'raw_reviews' table.
    This is the longest-running step and should only be run once.
    """
    print("\n--- STAGE 1: Staging Raw Data ---")
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
            
            sentiments = merged_chunk_df['text'].astype(str).apply(lambda text: TextBlob(text).sentiment)
            merged_chunk_df['sentiment'] = sentiments.apply(lambda s: 'Positive' if s.polarity > 0.1 else ('Negative' if s.polarity < -0.1 else 'Neutral'))
            merged_chunk_df['text_polarity'] = sentiments.apply(lambda s: s.polarity)
            merged_chunk_df['timestamp'] = pd.to_datetime(merged_chunk_df['timestamp'], unit='s', errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S')

            final_columns = ['parent_asin', 'product_title', 'category', 'image_urls', 'rating', 'text', 'sentiment', 'text_polarity', 'timestamp']
            
            write_mode = 'replace' if is_first_write and chunk_num == 1 else 'append'
            merged_chunk_df[final_columns].to_sql('raw_reviews', conn, if_exists=write_mode, index=False)
            
            is_first_write = False
            chunk_num += 1
        print(f"✅ Finished staging '{category_name}' in {time.time() - phase_start_time:.2f} seconds.")
    print("✅ Raw data staging complete.")

def create_products_table(conn):
    """Creates the aggregated 'products' table for the main gallery."""
    print("\n--- Building: products table ---")
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS products")
    cursor.execute("""
        CREATE TABLE products AS
        SELECT
            parent_asin,
            FIRST_VALUE(product_title) OVER (PARTITION BY parent_asin ORDER BY rowid) as product_title,
            FIRST_VALUE(image_urls) OVER (PARTITION BY parent_asin ORDER BY rowid) as image_urls,
            FIRST_VALUE(category) OVER (PARTITION BY parent_asin ORDER BY rowid) as category,
            AVG(rating) as average_rating,
            COUNT(rating) as review_count
        FROM raw_reviews GROUP BY parent_asin;
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_products_category ON products (category);")
    conn.commit()
    print("✅ 'products' table created.")

def create_reviews_table(conn):
    """Creates the lean 'reviews' table for paginated display."""
    print("\n--- Building: reviews table ---")
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS reviews")
    cursor.execute("""
        CREATE TABLE reviews AS
        SELECT parent_asin, rating, sentiment, text FROM raw_reviews;
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_reviews_parent_asin ON reviews (parent_asin);")
    conn.commit()
    print("✅ 'reviews' table created.")

def create_discrepancy_table(conn):
    """Creates the lightweight table for the discrepancy plot."""
    print("\n--- Building: discrepancy_data table ---")
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS discrepancy_data")
    cursor.execute("""
        CREATE TABLE discrepancy_data AS
        SELECT parent_asin, rating, text_polarity FROM raw_reviews;
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_discrepancy_parent_asin ON discrepancy_data (parent_asin);")
    conn.commit()
    print("✅ 'discrepancy_data' table created.")

def create_rating_distribution_table(conn):
    """(NEW) Pre-computes the count of each rating (1-5 stars) for every product."""
    print("\n--- Building: rating_distribution table ---")
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS rating_distribution")
    cursor.execute("""
        CREATE TABLE rating_distribution AS
        SELECT
            parent_asin,
            COUNT(CASE WHEN rating = 1.0 THEN 1 END) as '1_star',
            COUNT(CASE WHEN rating = 2.0 THEN 1 END) as '2_star',
            COUNT(CASE WHEN rating = 3.0 THEN 1 END) as '3_star',
            COUNT(CASE WHEN rating = 4.0 THEN 1 END) as '4_star',
            COUNT(CASE WHEN rating = 5.0 THEN 1 END) as '5_star'
        FROM raw_reviews
        GROUP BY parent_asin;
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_rating_dist_parent_asin ON rating_distribution (parent_asin);")
    conn.commit()
    print("✅ 'rating_distribution' table created.")

def create_aspects_table(conn, nlp):
    """Performs the intensive aspect analysis and saves the results."""
    print("\n--- Building: aspect_sentiments table ---")
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='aspect_sentiments'")
    if cursor.fetchone():
        print("✅ 'aspect_sentiments' table already exists. Skipping.")
        return
        
    print("Pre-computing all aspect sentiments (this will take a very long time)...")
    df_for_aspects = pd.read_sql("SELECT parent_asin, text FROM raw_reviews", conn)
    aspect_results = []
    
    for parent_asin, group in df_for_aspects.groupby('parent_asin'):
        print(f"  - Analyzing aspects for product: {parent_asin}...")
        aspect_sentiments = {aspect: {'positive': 0, 'negative': 0, 'neutral': 0} for aspect in ALL_ASPECTS}
        for review_text in group['text'].dropna():
            try:
                doc = nlp(review_text)
                for sentence in doc.sents:
                    for aspect in ALL_ASPECTS:
                        if f' {aspect.lower()} ' in f' {sentence.text.lower()} ':
                            sentiment = TextBlob(sentence.text).sentiment.polarity
                            if sentiment > 0.1: aspect_sentiments[aspect]['positive'] += 1
                            elif sentiment < -0.1: aspect_sentiments[aspect]['negative'] += 1
                            else: aspect_sentiments[aspect]['neutral'] += 1
            except Exception:
                continue
        
        for aspect, scores in aspect_sentiments.items():
            if scores['positive'] > 0 or scores['negative'] > 0 or scores['neutral'] > 0:
                aspect_results.append({
                    'parent_asin': parent_asin,
                    'aspect': aspect,
                    'positive': scores['positive'],
                    'negative': scores['negative'],
                    'neutral': scores['neutral']
                })
    
    pd.DataFrame(aspect_results).to_sql('aspect_sentiments', conn, if_exists='replace', index=False)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_aspects_parent_asin ON aspect_sentiments (parent_asin);")
    conn.commit()
    print("✅ 'aspect_sentiments' table created.")

# --- Main Execution Controller ---
def main():
    start_time = time.time()
    
    conn = sqlite3.connect(OUTPUT_DB_PATH)
    nlp = None
    
    # --- CONTROL PANEL ---
    # Comment or un-comment the steps you want to run.
    
    # STEP 1: Stage raw data. This is the longest step. Run it only once.
    print("--- Checking Step 1: Staging Raw Data ---")
    stage_raw_data(conn, '/kaggle/input/' + KAGGLE_INPUT_DIR)
    
    # STEP 2: Create the main aggregated tables. These are fast and can be re-run.
    print("\n--- Checking Step 2: Creating Core Tables ---")
    create_products_table(conn)
    create_reviews_table(conn)
    create_discrepancy_table(conn)
    create_rating_distribution_table(conn) # <-- Includes the new rating distribution table
    
    # STEP 3: Run the most expensive analysis. Run it only once.
    print("\n--- Checking Step 3: Creating Aspect Sentiments Table ---")
    print("Loading spaCy model for aspect analysis...")
    nlp = spacy.load("en_core_web_sm")
    create_aspects_table(conn, nlp)
    
    # STEP 4: Clean up the raw data table to save space (optional, but recommended for the final DB).
    print("\n--- Checking Step 4: Final Cleanup ---")
    conn.execute("DROP TABLE IF EXISTS raw_reviews;")
    conn.execute("VACUUM;")
    conn.commit()
    print("✅ Cleanup complete.")
    
    conn.close()

    end_time = time.time()
    print(f"\n✅ Database build complete in {end_time - start_time:.2f} seconds.")

if __name__ == '__main__':
    main()
