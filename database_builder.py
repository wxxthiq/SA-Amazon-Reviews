# database_builder.py (V8.4 - Kaggle Version)
import pandas as pd
import duckdb
import json
import os
import time
from tqdm.auto import tqdm
import torch
from transformers import pipeline

# --- Configuration for Kaggle Notebooks ---
# Assumes your dataset slug is 'mcauley-jsonl'
KAGGLE_INPUT_DIR = '/kaggle/input/mcauley-jsonl/' 
OUTPUT_FOLDER = '/kaggle/working/'
OUTPUT_DB_PATH = os.path.join(OUTPUT_FOLDER, 'amazon_reviews_final.duckdb')

# --- UPDATE: Remove .gz extension from filenames ---
CATEGORIES_TO_PROCESS = {
    'All Beauty': ('All_Beauty.jsonl/All_Beauty.jsonl'  , 'meta_All_Beauty.jsonl/meta_All_Beauty.jsonl')
}
TOP_N_PRODUCTS = 100
SENTIMENT_BATCH_SIZE = 512

# --- Helper Functions ---
# --- UPDATE: Reverted to a standard file reader ---
def parse_jsonl_to_df(path):
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
    if not isinstance(image_data, list): return None
    urls = [img.get('hi_res') or img.get('large') for img in image_data if isinstance(img, dict)]
    return ",".join(filter(None, urls)) if urls else None

# --- Main Build Process ---
def main():
    start_time = time.time()
    print(f"--- Starting Database Build (V8.4 - Kaggle) ---")

    # --- STAGE 0: Initialize Models ---
    print("\n--- STAGE 0: Initializing Models ---")
    device = 0 if torch.cuda.is_available() else -1
    if device == 0: print("✅ GPU detected.")
    else: print("⚠️ WARNING: No GPU detected (will be slow).")

    model_name = "cardiffnlp/twitter-roberta-base-sentiment"
    sentiment_pipeline = pipeline("sentiment-analysis", model=model_name, device=device)
    print("✅ Models loaded successfully.")

    # --- Database Setup (Enriched Products Table) ---
    conn = duckdb.connect(database=OUTPUT_DB_PATH, read_only=False)
    conn.execute("""
    CREATE OR REPLACE TABLE products (
        parent_asin VARCHAR PRIMARY KEY, product_title VARCHAR, category VARCHAR, 
        store VARCHAR, image_urls VARCHAR, features VARCHAR, description VARCHAR, 
        details VARCHAR, average_rating FLOAT, review_count INTEGER
    );
    """)
    conn.execute("""
    CREATE OR REPLACE TABLE reviews (
        review_id VARCHAR PRIMARY KEY, parent_asin VARCHAR, rating INTEGER, 
        review_title VARCHAR, text VARCHAR, date DATE, helpful_vote INTEGER, 
        verified_purchase BOOLEAN, sentiment VARCHAR, sentiment_score FLOAT
    );
    """)
    print("✅ Enriched 2-Table database schema created successfully.")

    master_products_list = []
    master_reviews_list = []

    # --- Main Loop ---
    for category_name, (review_file, meta_file) in CATEGORIES_TO_PROCESS.items():
        phase_start_time = time.time()
        print(f"\n\n--- Processing Category: {category_name} ---")

        reviews_df = parse_jsonl_to_df(os.path.join(KAGGLE_INPUT_DIR, review_file))
        if reviews_df is None or reviews_df.empty: continue
        
        meta_df = parse_jsonl_to_df(os.path.join(KAGGLE_INPUT_DIR, meta_file))
        if meta_df is None or meta_df.empty: continue

        top_products = reviews_df['parent_asin'].value_counts().nlargest(TOP_N_PRODUCTS).index.tolist()
        reviews_df = reviews_df[reviews_df['parent_asin'].isin(top_products)].copy()
        reviews_df.dropna(subset=['text'], inplace=True)
        reviews_df['text'] = reviews_df['text'].astype(str)
        reviews_df['review_id'] = reviews_df['parent_asin'] + '-' + reviews_df.index.astype(str)
        print(f"  - Loaded and filtered {len(reviews_df)} reviews.")

        # --- PHASE 1: Perform Sentiment Analysis ---
        print("\n--- PHASE 1: Performing Sentiment Analysis ---")
        batch_size_reviews = 10000 
        num_batches = (len(reviews_df) // batch_size_reviews) + 1
        aggregated_sentiments_list = []
        print(f"  - Processing {len(reviews_df)} reviews in {num_batches} batches...")

        for i in tqdm(range(num_batches), desc="  - Analyzing Batches"):
            start_index = i * batch_size_reviews
            end_index = start_index + batch_size_reviews
            review_batch_df = reviews_df.iloc[start_index:end_index]

            if review_batch_df.empty: continue

            sentence_map = []
            for row in review_batch_df.itertuples():
                sentences = [s.strip() for s in row.text.split('.') if s.strip()]
                for sentence in sentences:
                    sentence_map.append({'review_id': row.review_id, 'sentence': sentence})

            if not sentence_map: continue
            
            sentences_df = pd.DataFrame(sentence_map)
            all_sentences = sentences_df['sentence'].tolist()
            sentiment_results = sentiment_pipeline(all_sentences, batch_size=SENTIMENT_BATCH_SIZE, truncation=True)
            
            results_df = pd.DataFrame(sentiment_results)
            sentences_df = pd.concat([sentences_df.reset_index(drop=True), results_df], axis=1)

            label_map = {
                'LABEL_2': 1, 'Positive': 1, 'LABEL_1': 0, 
                'Neutral': 0, 'LABEL_0': -1, 'Negative': -1
            }
            sentences_df['numeric_score'] = sentences_df['label'].map(label_map).fillna(0) * sentences_df['score']
            
            batch_aggregated_sentiments = sentences_df.groupby('review_id')['numeric_score'].mean().reset_index()
            aggregated_sentiments_list.append(batch_aggregated_sentiments)

        if aggregated_sentiments_list:
            aggregated_sentiments = pd.concat(aggregated_sentiments_list, ignore_index=True)
            def score_to_label(score):
                if score > 0.1: return 'Positive'
                elif score < -0.1: return 'Negative'
                return 'Neutral'
            aggregated_sentiments['final_label'] = aggregated_sentiments['numeric_score'].apply(score_to_label)
            reviews_df = pd.merge(reviews_df, aggregated_sentiments, on='review_id', how='left')
            reviews_df.rename(columns={'final_label': 'sentiment', 'numeric_score': 'sentiment_score'}, inplace=True)
        
        reviews_df['sentiment'] = reviews_df['sentiment'].fillna('Neutral')
        reviews_df['sentiment_score'] = reviews_df['sentiment_score'].fillna(0.0)
        print("  - Sentiment analysis complete.")
        
        # --- PHASE 2: Preparing DataFrames for Consolidation ---
        print("\n--- PHASE 2: Preparing DataFrames for Ingestion ---")
        meta_df.rename(columns={'title': 'product_title'}, inplace=True)
        meta_df['image_urls'] = meta_df['images'].apply(extract_and_join_image_urls)
        
        for col in ['features', 'description', 'details']:
            if col in meta_df.columns:
                meta_df[col] = meta_df[col].apply(lambda x: json.dumps(x) if x else None)

        product_aggs = reviews_df.groupby('parent_asin').agg(average_rating=('rating', 'mean'), review_count=('review_id', 'count')).reset_index()
        
        # Select the new metadata columns
        meta_cols_to_select = ['parent_asin', 'product_title', 'store', 'image_urls', 'features', 'description', 'details']
        
        # Ensure all required meta columns exist before selecting them
        final_meta_cols = [col for col in meta_cols_to_select if col in meta_df.columns]
        
        products_final_df = pd.merge(meta_df[final_meta_cols], product_aggs, on='parent_asin', how='inner')
        products_final_df['category'] = category_name
        master_products_list.append(products_final_df)

        reviews_df['date'] = pd.to_datetime(reviews_df['timestamp'], unit='ms', errors='coerce').dt.date
        reviews_df['helpful_vote'] = pd.to_numeric(reviews_df['helpful_vote'], errors='coerce').fillna(0).astype(int)
        
        # --- FIX: Ensure all necessary columns are selected for the final reviews DataFrame ---
        final_review_cols = [
            'review_id', 'parent_asin', 'rating', 'title', 'text', 
            'date', 'helpful_vote', 'verified_purchase', 'sentiment', 'sentiment_score'
        ]
        reviews_final_df = reviews_df[final_review_cols].rename(columns={'title': 'review_title'})
        master_reviews_list.append(reviews_final_df)
        
        print(f"✅ Finished processing '{category_name}' in {time.time() - phase_start_time:.2f} seconds.")

    # --- FINAL INGESTION ---
    print("\n\n--- Final Ingestion: Writing all data to database ---")

    if master_products_list:
        final_products_df = pd.concat(master_products_list, ignore_index=True)
        conn.register('products_view', final_products_df)
        
        # --- FIX: Explicitly list the columns to ensure correct mapping ---
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

    if master_reviews_list:
        final_reviews_df = pd.concat(master_reviews_list, ignore_index=True)
        conn.register('reviews_view', final_reviews_df)
        conn.execute("INSERT INTO reviews SELECT * FROM reviews_view;")
        conn.unregister('reviews_view')
        print(f"  - Inserted {len(final_reviews_df)} records into 'reviews'.")

    conn.close()
    end_time = time.time()
    print(f"\n\n✅✅ Database build complete in {end_time - start_time:.2f} seconds.")
    print(f"Database file created at: {OUTPUT_DB_PATH}")

# --- Run the main function ---
if __name__ == '__main__':
    main()
