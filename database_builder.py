# database_builder.py (UPDATED for CSV files)

import pandas as pd
import sqlite3
import gzip
import json
from textblob import TextBlob
import time

# --- Configuration ---

# 1. The script is now pre-configured for your chosen CSV categories.
FILES_TO_PROCESS = [
    'fashion.csv.gz',
    'electronics.csv.gz',
    'home_and_kitchen.csv.gz'
]

# 2. This should match the DATABASE_PATH in your main app.py
OUTPUT_DB_PATH = 'amazon_fashion.db'

# 3. Define the column names for the CSV files, as they have no header row.
#    This is based on the schema from the dataset's GitHub page.
CSV_COLUMN_NAMES = [
    'rating', 'title', 'text', 'images', 'asin', 'parent_asin',
    'user_id', 'timestamp', 'helpful_vote', 'verified_purchase'
]

# --- End Configuration ---


def parse_csv_gz_to_df(path):
    """
    Reads a gzipped CSV file and returns a DataFrame.
    """
    # Extracts category name from filename, e.g., "home_and_kitchen.csv.gz" -> "Home And Kitchen"
    category_name = path.split('.csv.gz')[0].replace('_', ' ').title()
    print(f"Processing {category_name}...")

    # Use pandas to read the gzipped CSV directly
    df = pd.read_csv(path, compression='gzip', header=None, names=CSV_COLUMN_NAMES)
    
    # Add the category name to each review record
    df['category'] = category_name
    return df

def analyze_sentiment(text):
    """Performs sentiment analysis on a text string."""
    if not isinstance(text, str):
        return 'Neutral'
    
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.1:
        return 'Positive'
    elif polarity < -0.1:
        return 'Negative'
    else:
        return 'Neutral'

def main():
    start_time = time.time()
    
    all_dfs = []
    for file_path in FILES_TO_PROCESS:
        try:
            df = parse_csv_gz_to_df(file_path)
            all_dfs.append(df)
        except FileNotFoundError:
            print(f"ERROR: File not found: {file_path}. Please make sure it is in the same directory.")
            continue
    
    if not all_dfs:
        print("CRITICAL ERROR: No data files were processed. Exiting.")
        return

    print("\nCombining all datasets...")
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    # The CSV columns are already well-named, so less renaming is needed.
    # We just need to handle 'title' -> 'product_title' and 'images' -> 'image_url'
    combined_df = combined_df.rename(columns={
        'title': 'product_title',
        'images': 'image_url'
    })
    
    print("Performing sentiment analysis on all reviews (this will take several minutes)...")
    combined_df['sentiment'] = combined_df['text'].apply(analyze_sentiment)
    
    print("Converting timestamps...")
    combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'], unit='s')
    
    print("Calculating product-level average ratings and review counts...")
    aggregates = combined_df.groupby('parent_asin').agg(
        review_count=('rating', 'count'),
        average_rating=('rating', 'mean')
    ).reset_index()
    
    final_df = pd.merge(combined_df, aggregates, on='parent_asin', how='left')
    
    # Ensure product_title is not null
    final_df['product_title'].fillna("Product " + final_df['parent_asin'], inplace=True)
        
    # Select and reorder columns to match the app's expectations
    final_columns = [
        'parent_asin', 'product_title', 'category', 'rating', 'text', 
        'sentiment', 'timestamp', 'average_rating', 'review_count', 'image_url'
    ]
    final_columns_exist = [col for col in final_columns if col in final_df.columns]
    final_df = final_df[final_columns_exist]

    print(f"Saving combined data to SQLite database: {OUTPUT_DB_PATH}")
    conn = sqlite3.connect(OUTPUT_DB_PATH)
    final_df.to_sql('reviews', conn, if_exists='replace', index=False)
    conn.close()
    
    end_time = time.time()
    print("\n-------------------------")
    print("✅ Database build complete!")
    print("-------------------------")
    print(f"Total rows processed: {len(final_df):,}")
    print(f"Total unique products: {final_df['parent_asin'].nunique():,}")
    print(f"Categories included: {final_df['category'].unique().tolist()}")
    print(f"Total time: {end_time - start_time:.2f} seconds")

if __name__ == '__main__':
    main()