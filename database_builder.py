import os
# ==============================================================================
# 1. INSTALL REQUIRED LIBRARIES
# ==============================================================================
# This section ensures the correct versions of libraries are installed in the
# Kaggle environment for both sentiment analysis and aspect-based analysis.
# ==============================================================================
print("--- Installing required libraries ---")
# Uninstall potentially conflicting libraries
os.system('pip uninstall -y pytorch-lightning kaggle-environments torchaudio torchvision fastai tensorflow keras tensorflow-decision-forests dopamine-rl sentence-transformers')

# FIX: Install a new set of specific, compatible versions in a single command to resolve the torch/accelerate conflict.
os.system('pip install -q "pyabsa==1.16.28" torch==2.1.0 transformers==4.30.0 accelerate==0.25.0 tokenizers==0.13.3')
print("✅ Libraries installed.")
# ==============================================================================
# CELL 2: ANALYSIS SCRIPT (ATEPC using pyabsa v1.16.28)
# ==============================================================================
# IMPORTANT: Run this cell only *after* running the setup cell above and
# waiting for the kernel to restart.
# ==============================================================================
import pandas as pd
import torch
# FIX: Import the correct 'ATEPCCheckpointManager' for inference as per v1.16.28 documentation.
from pyabsa.functional import ATEPCCheckpointManager
from tqdm.auto import tqdm
import json
import os
import gzip
import time
from transformers import pipeline
import duckdb

# Set environment variable to prevent potential protobuf conflicts
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
print("✅ Libraries imported successfully after restart.")



# --- Configuration for Kaggle Notebooks ---
KAGGLE_INPUT_DIR = '/kaggle/input/mcauley-jsonl/'
OUTPUT_FOLDER = '/kaggle/working/'
OUTPUT_DB_PATH = os.path.join(OUTPUT_FOLDER, 'amazon_reviews_final.duckdb')

# Define categories and product limits
CATEGORIES_TO_PROCESS = {
    'All Beauty': ('All_Beauty.jsonl/All_Beauty.jsonl', 'meta_All_Beauty.jsonl/meta_All_Beauty.jsonl')
}
TOP_N_PRODUCTS = 50 # 50 products per category
SENTIMENT_BATCH_SIZE = 256 # Batch size for roberta sentiment model
ASPECT_BATCH_SIZE = 128    # Batch size for pyabsa aspect model


# --- Helper Functions ---
def parse_jsonl_to_df(path):
    print(f"  Reading data from: {os.path.basename(path)}")
    data = []
    try:
        # Assuming the files are not gzipped based on the provided path
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
        return pd.DataFrame.from_records(data)
    except Exception as e:
        print(f"    ❌ ERROR reading file {path}: {e}")
        return None

def extract_and_join_image_urls(image_data):
    if not isinstance(image_data, list): return None
    urls = [img.get('hi_res') or img.get('large') for img in image_data if isinstance(img, dict)]
    return ",".join(filter(None, urls)) if urls else None

# --- Main Build Process ---
def main():
    start_time = time.time()
    print(f"--- Starting Database Build (V9.2 - Robust Ingestion) ---")

    # --- STAGE 0: Initialize Models ---
    print("\n--- STAGE 0: Initializing Models ---")
    device = 0 if torch.cuda.is_available() else -1
    if device == 0: print("✅ GPU detected.")
    else: print("⚠️ WARNING: No GPU detected (will be slow).")

    # Model for overall review sentiment
    # Explicitly add max_length and padding to the pipeline to handle long reviews.
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment",
        device=device,
        truncation=True,
        padding=True,
        max_length=512
    )
    print("✅ Roberta sentiment model loaded.")

    # Model for Aspect Term Extraction and Polarity Classification (ATEPC)
    aspect_extractor = ATEPCCheckpointManager.get_aspect_extractor(checkpoint='english', auto_device=True)
    print("✅ PyABSA ATEPC model loaded.")


    # --- Database Setup (Enriched 3-Table Schema) ---
    conn = duckdb.connect(database=OUTPUT_DB_PATH, read_only=False)
    # Table 1: Products
    conn.execute("""
    CREATE OR REPLACE TABLE products (
        parent_asin VARCHAR PRIMARY KEY, product_title VARCHAR, category VARCHAR,
        store VARCHAR, image_urls VARCHAR, features VARCHAR, description VARCHAR,
        details VARCHAR, average_rating FLOAT, review_count INTEGER
    );
    """)
    # Table 2: Reviews
    conn.execute("""
    CREATE OR REPLACE TABLE reviews (
        review_id VARCHAR PRIMARY KEY, parent_asin VARCHAR, rating INTEGER,
        review_title VARCHAR, text VARCHAR, date DATE, helpful_vote INTEGER,
        verified_purchase BOOLEAN, sentiment VARCHAR, sentiment_score FLOAT
    );
    """)
    # Table 3: Aspects (NEW)
    conn.execute("""
    CREATE OR REPLACE TABLE aspects (
        aspect_id VARCHAR PRIMARY KEY,
        review_id VARCHAR,
        parent_asin VARCHAR,
        aspect VARCHAR,
        sentiment VARCHAR
    );
    """)
    print("✅ Enriched 3-Table database schema created successfully.")

    master_products_list = []
    master_reviews_list = []
    master_aspects_list = [] # New list for aspect data

    # --- Main Loop ---
    for category_name, (review_file, meta_file) in CATEGORIES_TO_PROCESS.items():
        phase_start_time = time.time()
        print(f"\n\n--- Processing Category: {category_name} ---")

        # Load and filter reviews
        reviews_df = parse_jsonl_to_df(os.path.join(KAGGLE_INPUT_DIR, review_file))
        if reviews_df is None or reviews_df.empty: continue

        meta_df = parse_jsonl_to_df(os.path.join(KAGGLE_INPUT_DIR, meta_file))
        if meta_df is None or meta_df.empty: continue

        top_products = reviews_df['parent_asin'].value_counts().nlargest(TOP_N_PRODUCTS).index.tolist()
        reviews_df = reviews_df[reviews_df['parent_asin'].isin(top_products)].copy()
        reviews_df.dropna(subset=['text'], inplace=True)
        reviews_df['text'] = reviews_df['text'].astype(str)
        reviews_df['review_id'] = reviews_df['parent_asin'] + '-' + reviews_df.index.astype(str)
        print(f"  - Loaded and filtered {len(reviews_df)} reviews for {len(top_products)} products.")

        # --- PHASE 1: Perform Overall Sentiment Analysis ---
        print("\n--- PHASE 1: Performing Overall Sentiment Analysis ---")
        sentiments_data = []
        for i in tqdm(range(0, len(reviews_df), SENTIMENT_BATCH_SIZE), desc="  - Analyzing Sentiment"):
            batch_texts = reviews_df['text'].iloc[i:i+SENTIMENT_BATCH_SIZE].tolist()
            sentiments_data.extend(sentiment_pipeline(batch_texts))

        label_map = {'LABEL_2': (1, 'Positive'), 'LABEL_1': (0, 'Neutral'), 'LABEL_0': (-1, 'Negative')}
        reviews_df['sentiment_score'] = [s['score'] * label_map.get(s['label'], (0, 'Neutral'))[0] for s in sentiments_data]
        reviews_df['sentiment'] = [label_map.get(s['label'], (0, 'Neutral'))[1] for s in sentiments_data]
        print("  - Overall sentiment analysis complete.")

        # --- PHASE 2: Perform Aspect-Based Sentiment Analysis (NEW) ---
        print("\n--- PHASE 2: Performing Aspect-Based Sentiment Analysis ---")
        aspect_sentiments = []
        review_ids_list = reviews_df['review_id'].tolist()
        parent_asins_list = reviews_df['parent_asin'].tolist()

        for i in tqdm(range(0, len(reviews_df), ASPECT_BATCH_SIZE), desc="  - Extracting Aspects"):
            batch_texts = reviews_df['text'].iloc[i:i+ASPECT_BATCH_SIZE].tolist()
            batch_ids = review_ids_list[i:i+ASPECT_BATCH_SIZE]
            batch_asins = parent_asins_list[i:i+ASPECT_BATCH_SIZE]

            results = aspect_extractor.extract_aspect(
                inference_source=batch_texts,
                save_result=False,
                print_result=False,
                pred_sentiment=True
            )

            for j, res in enumerate(results):
                review_id = batch_ids[j]
                parent_asin = batch_asins[j]
                aspect_count = 0
                for aspect, sentiment in zip(res['aspect'], res['sentiment']):
                    if aspect != 'None':
                        aspect_sentiments.append({
                            'aspect_id': f"{review_id}-{aspect_count}",
                            'review_id': review_id,
                            'parent_asin': parent_asin,
                            'aspect': aspect.lower(),
                            'sentiment': sentiment.capitalize()
                        })
                        aspect_count += 1

        if aspect_sentiments:
            category_aspects_df = pd.DataFrame(aspect_sentiments)
            master_aspects_list.append(category_aspects_df)
            print(f"  - Extracted {len(category_aspects_df)} aspect-sentiment pairs.")
        else:
            print("  - No aspects extracted for this category.")


        # --- PHASE 3: Preparing DataFrames for Consolidation ---
        print("\n--- PHASE 3: Preparing DataFrames for Ingestion ---")
        # Prepare products data
        meta_df.rename(columns={'title': 'product_title'}, inplace=True)
        meta_df['image_urls'] = meta_df['images'].apply(extract_and_join_image_urls)
        for col in ['features', 'description', 'details']:
            if col in meta_df.columns:
                meta_df[col] = meta_df[col].apply(lambda x: json.dumps(x) if x else None)
        product_aggs = reviews_df.groupby('parent_asin').agg(average_rating=('rating', 'mean'), review_count=('review_id', 'count')).reset_index()
        meta_cols = [col for col in ['parent_asin', 'product_title', 'store', 'image_urls', 'features', 'description', 'details'] if col in meta_df.columns]
        products_final_df = pd.merge(meta_df[meta_cols], product_aggs, on='parent_asin', how='inner')
        products_final_df['category'] = category_name
        master_products_list.append(products_final_df)

        # Prepare reviews data
        reviews_df['date'] = pd.to_datetime(reviews_df['timestamp'], unit='ms', errors='coerce').dt.date
        reviews_df['helpful_vote'] = pd.to_numeric(reviews_df['helpful_vote'], errors='coerce').fillna(0).astype(int)
        final_review_cols = ['review_id', 'parent_asin', 'rating', 'title', 'text', 'date', 'helpful_vote', 'verified_purchase', 'sentiment', 'sentiment_score']
        reviews_final_df = reviews_df[final_review_cols].rename(columns={'title': 'review_title'})
        master_reviews_list.append(reviews_final_df)

        print(f"✅ Finished processing '{category_name}' in {time.time() - phase_start_time:.2f} seconds.")

    # --- FINAL INGESTION ---
    print("\n\n--- Final Ingestion: Writing all data to database ---")

    # Ingest Products
    if master_products_list:
        final_products_df = pd.concat(master_products_list, ignore_index=True)
        conn.register('products_view', final_products_df)
        # FIX: Use an explicit INSERT statement to avoid column order errors.
        conn.execute("""
            INSERT INTO products (
                parent_asin, product_title, category, store, image_urls,
                features, description, details, average_rating, review_count
            )
            SELECT
                parent_asin, product_title, category, store, image_urls,
                features, description, details, average_rating, review_count
            FROM products_view;
        """)
        conn.unregister('products_view')
        print(f"  - Inserted {len(final_products_df)} records into 'products'.")

    # Ingest Reviews
    if master_reviews_list:
        final_reviews_df = pd.concat(master_reviews_list, ignore_index=True)
        conn.register('reviews_view', final_reviews_df)
        # FIX: Use an explicit INSERT statement for robustness.
        conn.execute("""
            INSERT INTO reviews (
                review_id, parent_asin, rating, review_title, text, date,
                helpful_vote, verified_purchase, sentiment, sentiment_score
            )
            SELECT
                review_id, parent_asin, rating, review_title, text, date,
                helpful_vote, verified_purchase, sentiment, sentiment_score
            FROM reviews_view;
        """)
        conn.unregister('reviews_view')
        print(f"  - Inserted {len(final_reviews_df)} records into 'reviews'.")

    # Ingest Aspects (NEW)
    if master_aspects_list:
        final_aspects_df = pd.concat(master_aspects_list, ignore_index=True)
        conn.register('aspects_view', final_aspects_df)
        # FIX: Use an explicit INSERT statement for robustness.
        conn.execute("""
            INSERT INTO aspects (
                aspect_id, review_id, parent_asin, aspect, sentiment
            )
            SELECT
                aspect_id, review_id, parent_asin, aspect, sentiment
            FROM aspects_view;
        """)
        conn.unregister('aspects_view')
        print(f"  - Inserted {len(final_aspects_df)} records into 'aspects'.")

    conn.close()
    end_time = time.time()
    print(f"\n\n✅✅ Database build complete in {end_time - start_time:.2f} seconds.")
    print(f"Database file created at: {OUTPUT_DB_PATH}")

# --- Run the main function ---
if __name__ == '__main__':
    main()
