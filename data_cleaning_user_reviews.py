import pandas as pd
import glob
import os

# --- Configuration ---
# Directory containing the raw user review CSV files
input_review_dir = "./supp_data2"
# Directory where cleaned review files will be saved
output_review_dir = "./processed_reviews"

# Ensure output directory exists
os.makedirs(output_review_dir, exist_ok=True)

# Process all CSV files in input directory
csv_files = glob.glob(os.path.join(input_review_dir, '*.csv'))

for file_path in csv_files:
    filename = os.path.basename(file_path)
    print(f"Processing file: {filename}")
    
    # Attempt to read the CSV, skipping malformed lines
    try:
        df = pd.read_csv(
            file_path,
            engine='python',
            on_bad_lines='skip',
            encoding='utf-8',
            dtype=str  # read all as strings
        )
    except Exception as e:
        print(f"[ERROR] Could not load {filename}: {e}. Skipping.")
        continue

    # Dynamically detect the three target columns
    cols = df.columns
    author_col  = next((c for c in cols if 'author' in c.lower()), None)
    vehicle_col = next((c for c in cols if 'vehicle' in c.lower()), None)
    rating_col  = next((c for c in cols if 'rating' in c.lower()), None)

    if not all([author_col, vehicle_col, rating_col]):
        print(f"[WARNING] Required columns not found in {filename}. Skipping.")
        continue

    # Subset and rename
    cleaned_df = df[[author_col, vehicle_col, rating_col]].copy()
    cleaned_df.columns = ['author', 'vehicle_title', 'rating']

    # Save cleaned CSV
    output_file = os.path.join(
        output_review_dir,
        filename.replace('.csv', '_cleaned.csv')
    )
    cleaned_df.to_csv(output_file, index=False)
    print(f"Saved cleaned data to {os.path.basename(output_file)}\n")
