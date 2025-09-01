import os
import pandas as pd
import torch
from pyabsa.functional import ATEPCCheckpointManager
from tqdm.auto import tqdm
import json
import time
from transformers import pipeline
import duckdb
import numpy as np
from collections import Counter

# ==============================================================================
# INSTALL REQUIRED LIBRARIES
# ==============================================================================
# This section ensures the correct versions of libraries are installed in the
# Kaggle environment for both sentiment analysis and aspect-based analysis.
# ==============================================================================
print("--- Installing required libraries ---")
# Uninstall potentially conflicting libraries
os.system('pip uninstall -y pytorch-lightning kaggle-environments torchaudio torchvision fastai tensorflow keras tensorflow-decision-forests dopamine-rl sentence-transformers')

# Install a new set of specific, compatible versions in a single command to resolve the torch/accelerate conflict.
os.system('pip install -q "pyabsa==1.16.28" torch==2.1.0 transformers==4.30.0 accelerate==0.25.0 tokenizers==0.13.3')
print("✅ Libraries installed.")


# ==============================================================================
# SCRIPT START
# ==============================================================================
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
print("✅ Libraries imported successfully after restart.")

# --- Configuration ---
REVIEWS_INPUT_DIR = '/kaggle/input/amazon-reviews/'
META_INPUT_DIR = '/kaggle/input/amazon-meta/'
OUTPUT_FOLDER = '/kaggle/working/'
OUTPUT_DB_PATH = os.path.join(OUTPUT_FOLDER, 'amazon_reviews_final.duckdb')

CATEGORIES_TO_PROCESS = [
    'All_Beauty.jsonl',
    'Electronics.jsonl',
    'Home_and_Kitchen.jsonl',
    'Books.jsonl',
    'Clothing_Shoes_and_Jewelry.jsonl',
    'Health_and_Personal_Care.jsonl',
    'Sports_and_Outdoors.jsonl',
    'Toys_and_Games.jsonl',
    'Tools_and_Home_Improvement.jsonl',
    'Musical_Instruments.jsonl',
]
PRODUCTS_PER_CATEGORY = 500
SENTIMENT_BATCH_SIZE = 256
ASPECT_BATCH_SIZE = 128

# --- Helper Functions ---
def extract_and_join_image_urls(image_data):
    if not isinstance(image_data, list): return None
    urls = [img.get('hi_res') or img.get('large') for img in image_data if isinstance(img, dict)]
    return ",".join(filter(None, urls)) if urls else None

def extract_and_join_video_urls(video_data):
    if not isinstance(video_data, list): return None
    urls = [vid.get('url') for vid in video_data if isinstance(vid, dict)]
    return ",".join(filter(None, urls)) if urls else None

def safe_json_dumps(obj):
    if isinstance(obj, (list, dict)):
        if not obj: return None
    elif pd.isna(obj):
        return None
    return json.dumps(obj)


# --- Main Build Process ---
def main():
    start_time = time.time()
    print(f"--- Starting Database Build (Final Memory-Efficient Version) ---")

    # --- STAGE 0: Initialize Models ---
    print("\n--- STAGE 0: Initializing Models ---")
    device = 0 if torch.cuda.is_available() else -1
    if device == 0: print("✅ GPU detected.")
    else: print("⚠️ WARNING: No GPU detected (will be slow).")
    sentiment_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment", device=device, truncation=True, padding=True, max_length=512)
    print("✅ Roberta sentiment model loaded.")
    aspect_extractor = ATEPCCheckpointManager.get_aspect_extractor(checkpoint='english', auto_device=True)
    print("✅ PyABSA ATEPC model loaded.")

    # --- Database Schema ---
    conn = duckdb.connect(database=OUTPUT_DB_PATH, read_only=False)
    conn.execute("CREATE OR REPLACE TABLE products (parent_asin VARCHAR PRIMARY KEY, product_title VARCHAR, category VARCHAR, store VARCHAR, price VARCHAR, image_urls VARCHAR, videos VARCHAR, features VARCHAR, description VARCHAR, details VARCHAR, average_rating FLOAT, review_count INTEGER);")
    conn.execute("CREATE OR REPLACE TABLE reviews (review_id VARCHAR PRIMARY KEY, parent_asin VARCHAR, rating INTEGER, review_title VARCHAR, text VARCHAR, date DATE, helpful_vote INTEGER, verified_purchase BOOLEAN, videos VARCHAR, sentiment VARCHAR, sentiment_score FLOAT);")
    conn.execute("CREATE OR REPLACE TABLE aspects (aspect_id VARCHAR PRIMARY KEY, review_id VARCHAR, parent_asin VARCHAR, aspect VARCHAR, sentiment VARCHAR);")
    print("✅ Enriched database schema created successfully.")

    master_products_list = []
    master_reviews_list = []
    master_aspects_list = []

    # --- Main Loop ---
    for review_file_name in CATEGORIES_TO_PROCESS:
        phase_start_time = time.time()
        category_name = review_file_name.replace('.jsonl', '')
        meta_file_name = f"meta_{category_name}.jsonl"
        print(f"\n\n--- Processing Category: {category_name} ---")

        # --- STAGE 1: Memory-Efficient Two-Pass Subsampling ---
        print("\n--- STAGE 1: Performing Memory-Efficient Stratified Product Subsampling ---")
        review_file_path = os.path.join(REVIEWS_INPUT_DIR, review_file_name, review_file_name)
        if not os.path.exists(review_file_path):
            print(f"  - ⚠️ Skipping category {category_name} - review file not found.")
            continue
        print("  - Pass 1: Scanning for product review counts...")
        product_counts = Counter()
        with open(review_file_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="    Scanning reviews"):
                try:
                    review = json.loads(line)
                    if 'parent_asin' in review:
                        product_counts[review['parent_asin']] += 1
                except json.JSONDecodeError: continue
        if not product_counts:
            print(f"  - ⚠️ Skipping category {category_name} - no valid reviews found.")
            continue
        product_review_counts = pd.DataFrame(product_counts.items(), columns=['parent_asin', 'review_count'])
        print("  - Performing stratified sampling on products...")
        strata_labels = ["Long-Tail", "Mid-Range", "Popular", "Blockbuster"]
        try:
            product_review_counts['stratum'] = pd.qcut(product_review_counts['review_count'], q=4, labels=strata_labels)
        except ValueError:
            product_review_counts['stratum'] = "all"
        sampled_products_df = product_review_counts.groupby('stratum', group_keys=False).apply(
            lambda x: x.sample(n=min(len(x), int(len(x)/len(product_review_counts) * PRODUCTS_PER_CATEGORY)))
        ).head(PRODUCTS_PER_CATEGORY)
        sampled_product_asins = set(sampled_products_df['parent_asin'].tolist())
        print(f"  - Selected {len(sampled_product_asins)} products for ingestion.")
        print("  - Pass 2: Ingesting reviews for selected products...")
        sampled_reviews_data = []
        with open(review_file_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="    Ingesting reviews"):
                try:
                    review = json.loads(line)
                    if review.get('parent_asin') in sampled_product_asins:
                        sampled_reviews_data.append(review)
                except json.JSONDecodeError: continue
        reviews_df = pd.DataFrame(sampled_reviews_data)
        reviews_df.dropna(subset=['text'], inplace=True)
        reviews_df['text'] = reviews_df['text'].astype(str)
        reviews_df['review_id'] = reviews_df['parent_asin'] + '-' + reviews_df.index.astype(str)
        print(f"  - Successfully ingested {len(reviews_df)} reviews.")

        # --- STAGE 2 & 3: Sentiment and Aspect Analysis (No changes here) ---
        # ... (Identical to previous script) ...
        print("\n--- STAGE 2: Performing Overall Sentiment Analysis ---")
        sentiments_data = []
        for i in tqdm(range(0, len(reviews_df), SENTIMENT_BATCH_SIZE), desc="  - Analyzing Sentiment"):
            batch_texts = reviews_df['text'].iloc[i:i+SENTIMENT_BATCH_SIZE].tolist()
            sentiments_data.extend(sentiment_pipeline(batch_texts))
        label_map = {'LABEL_2': (1, 'Positive'), 'LABEL_1': (0, 'Neutral'), 'LABEL_0': (-1, 'Negative')}
        reviews_df['sentiment_score'] = [s['score'] * label_map.get(s['label'], (0, 'Neutral'))[0] for s in sentiments_data]
        reviews_df['sentiment'] = [label_map.get(s['label'], (0, 'Neutral'))[1] for s in sentiments_data]
        print("  - Overall sentiment analysis complete.")
        print("\n--- STAGE 3: Performing Aspect-Based Sentiment Analysis ---")
        aspect_sentiments = []
        review_ids_list = reviews_df['review_id'].tolist()
        parent_asins_list = reviews_df['parent_asin'].tolist()
        for i in tqdm(range(0, len(reviews_df), ASPECT_BATCH_SIZE), desc="  - Extracting Aspects"):
            batch_texts = reviews_df['text'].iloc[i:i+ASPECT_BATCH_SIZE].tolist()
            batch_ids = review_ids_list[i:i+ASPECT_BATCH_SIZE]
            batch_asins = parent_asins_list[i:i+ASPECT_BATCH_SIZE]
            results = aspect_extractor.extract_aspect(inference_source=batch_texts, save_result=False, print_result=False, pred_sentiment=True)
            for j, res in enumerate(results):
                review_id = batch_ids[j]
                parent_asin = batch_asins[j]
                aspect_count = 0
                for aspect, sentiment in zip(res['aspect'], res['sentiment']):
                    if aspect != 'None':
                        aspect_sentiments.append({'aspect_id': f"{review_id}-{aspect_count}", 'review_id': review_id, 'parent_asin': parent_asin, 'aspect': aspect.lower(), 'sentiment': sentiment.capitalize()})
                        aspect_count += 1
        if aspect_sentiments:
            master_aspects_list.append(pd.DataFrame(aspect_sentiments))
            print(f"  - Extracted {len(pd.DataFrame(aspect_sentiments))} aspect-sentiment pairs.")

        # --- MODIFIED: STAGE 4: Memory-Efficient Metadata Loading ---
        print("\n--- STAGE 4: Preparing DataFrames for Ingestion ---")
        meta_file_path = os.path.join(META_INPUT_DIR, meta_file_name, meta_file_name)
        
        # --- Read metadata file line-by-line, only keeping data for sampled products ---
        print("  - Ingesting metadata for selected products...")
        sampled_meta_data = []
        with open(meta_file_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="    Scanning metadata"):
                try:
                    meta = json.loads(line)
                    if meta.get('parent_asin') in sampled_product_asins:
                        sampled_meta_data.append(meta)
                except json.JSONDecodeError:
                    continue
        
        if not sampled_meta_data:
            print(f"  - ⚠️ Skipping metadata for {category_name} - no matching ASINs found in meta file.")
            continue
            
        meta_df = pd.DataFrame(sampled_meta_data)
        print(f"  - Successfully ingested {len(meta_df)} metadata records.")

        # --- The rest of Stage 4 is now safe and fast ---
        meta_df.rename(columns={'title': 'product_title'}, inplace=True)
        meta_df['image_urls'] = meta_df['images'].apply(extract_and_join_image_urls)
        if 'videos' in meta_df.columns:
            meta_df['videos'] = meta_df['videos'].apply(extract_and_join_video_urls)
        for col in ['features', 'description', 'details']:
            if col in meta_df.columns:
                meta_df[col] = meta_df[col].apply(safe_json_dumps)
        product_aggs = reviews_df.groupby('parent_asin').agg(average_rating=('rating', 'mean'), review_count=('review_id', 'count')).reset_index()
        meta_cols = [col for col in ['parent_asin', 'product_title', 'store', 'price', 'image_urls', 'videos', 'features', 'description', 'details'] if col in meta_df.columns]
        products_final_df = pd.merge(meta_df[meta_cols], product_aggs, on='parent_asin', how='inner')
        products_final_df['category'] = category_name
        master_products_list.append(products_final_df)
        reviews_df['date'] = pd.to_datetime(reviews_df['timestamp'], unit='ms', errors='coerce').dt.date
        reviews_df['helpful_vote'] = pd.to_numeric(reviews_df['helpful_vote'], errors='coerce').fillna(0).astype(int)
        if 'videos' in reviews_df.columns:
            reviews_df['videos'] = reviews_df['videos'].apply(extract_and_join_video_urls)
        final_review_cols = ['review_id', 'parent_asin', 'rating', 'title', 'text', 'date', 'helpful_vote', 'verified_purchase', 'videos', 'sentiment', 'sentiment_score']
        existing_review_cols = [col for col in final_review_cols if col in reviews_df.columns]
        reviews_final_df = reviews_df[existing_review_cols].rename(columns={'title': 'review_title'})
        master_reviews_list.append(reviews_final_df)
        print(f"✅ Finished processing '{category_name}' in {time.time() - phase_start_time:.2f} seconds.")

    # --- FINAL INGESTION ---
    # ... (Identical to previous script) ...
    print("\n\n--- Final Ingestion: Writing all data to database ---")
    if master_products_list:
        final_products_df = pd.concat(master_products_list, ignore_index=True)
        for col in ['price', 'videos']:
            if col not in final_products_df.columns: final_products_df[col] = None
        conn.register('products_view', final_products_df)
        conn.execute("INSERT INTO products SELECT parent_asin, product_title, category, store, price, image_urls, videos, features, description, details, average_rating, review_count FROM products_view;")
        conn.unregister('products_view')
        print(f"  - Inserted {len(final_products_df)} records into 'products'.")
    if master_reviews_list:
        final_reviews_df = pd.concat(master_reviews_list, ignore_index=True)
        if 'videos' not in final_reviews_df.columns: final_reviews_df['videos'] = None
        conn.register('reviews_view', final_reviews_df)
        conn.execute("INSERT INTO reviews SELECT review_id, parent_asin, rating, review_title, text, date, helpful_vote, verified_purchase, videos, sentiment, sentiment_score FROM reviews_view;")
        conn.unregister('reviews_view')
        print(f"  - Inserted {len(final_reviews_df)} records into 'reviews'.")
    if master_aspects_list:
        final_aspects_df = pd.concat(master_aspects_list, ignore_index=True)
        conn.register('aspects_view', final_aspects_df)
        conn.execute("INSERT INTO aspects SELECT * FROM aspects_view;")
        conn.unregister('aspects_view')
        print(f"  - Inserted {len(final_aspects_df)} records into 'aspects'.")

    conn.close()
    end_time = time.time()
    print(f"\n\n✅✅ Database build complete in {end_time - start_time:.2f} seconds.")
    print(f"Database file created at: {OUTPUT_DB_PATH}")

# --- Run the main function ---
if __name__ == '__main__':
    main()
