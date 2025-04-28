import pandas as pd
import numpy as np # Needed for np.nan comparison if any
import os

# --- Configuration ---
# Directory where the input files are located
input_dir = r"C:\Users\rudyk\Downloads\proj_data"
# Input file name
input_file_name = "vehicles.csv"
# Directory where cleaned files will be saved
output_dir = "./cleaned_data" # Corrected variable name
# Output file name
output_file_name = "vehicles_cleaned.csv"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Construct full file paths
input_file_path = os.path.join(input_dir, input_file_name)
output_file_path = os.path.join(output_dir, output_file_name)

print(f"Input file: {input_file_path}")
print(f"Output directory for cleaned file: {output_dir}")
print(f"Output file will be: {output_file_path}")

# --- Data Loading ---
try:
    df = pd.read_csv(input_file_path)
    print(f"\n--- Successfully loaded data. Initial shape: {df.shape} ---")

    # --- Cleaning Steps ---

    # 1. Drop Columns
    columns_to_drop = ['county', 'size', 'url', 'region_url', 'image_url', 'description', 'VIN', 'id']
    # Ensure columns exist before dropping
    columns_to_drop = [col for col in columns_to_drop if col in df.columns]
    df.drop(columns=columns_to_drop, inplace=True)
    print(f"Step 1: Dropped columns: {columns_to_drop}. New shape: {df.shape}")

    # 2. Handle Missing Values (Drop Rows)
    # Drop rows where key identifiers or critical fields are missing
    critical_cols_dropna = ['year', 'odometer', 'lat', 'long', 'posting_date', 'manufacturer', 'model']
    initial_rows = df.shape[0]
    df.dropna(subset=critical_cols_dropna, inplace=True)
    rows_dropped = initial_rows - df.shape[0]
    print(f"Step 2: Dropped {rows_dropped} rows with missing values in {critical_cols_dropna}. New shape: {df.shape}")

    # 3. Filter Outliers / Unrealistic Values
    initial_rows = df.shape[0]
    # Price: Keep between $500 and $400,000
    df = df[(df['price'] >= 500) & (df['price'] <= 400000)]
    # Year: Keep >= 1960
    df = df[df['year'] >= 1960]
    # Odometer: Keep between 100 and 600,000 miles
    df = df[(df['odometer'] >= 100) & (df['odometer'] <= 600000)]
    rows_filtered = initial_rows - df.shape[0]
    print(f"Step 3: Filtered {rows_filtered} rows based on price, year, odometer ranges. New shape: {df.shape}")

    # 4. Handle Missing Values (Imputation)
    # Impute with 'unknown' for columns with many missing or diverse categories
    cols_impute_unknown = ['condition', 'cylinders', 'drive', 'type', 'paint_color']
    for col in cols_impute_unknown:
        if col in df.columns: # Check if column still exists
             df[col].fillna('unknown', inplace=True)
    print(f"Step 4a: Imputed NaNs with 'unknown' in {cols_impute_unknown}.")

    # Impute with mode for columns with fewer missing and clear dominant category
    cols_impute_mode = ['fuel', 'title_status', 'transmission']
    for col in cols_impute_mode:
         if col in df.columns: # Check if column still exists
            mode_val = df[col].mode()[0]
            df[col].fillna(mode_val, inplace=True)
    print(f"Step 4b: Imputed NaNs with mode in {cols_impute_mode}.")

    # 5. Convert Data Types
    # Convert Year to integer
    df['year'] = df['year'].astype(int)
    # Convert Odometer to integer
    df['odometer'] = df['odometer'].astype(int)
    # Convert posting_date to datetime (errors='coerce' will turn unparseable dates into NaT - Not a Time)
    df['posting_date'] = pd.to_datetime(df['posting_date'], errors='coerce', utc=True) # Added utc=True based on format
    # Check if any dates failed to parse
    failed_date_parses = df['posting_date'].isnull().sum()
    if failed_date_parses > 0:
        print(f"Warning: {failed_date_parses} entries in 'posting_date' could not be parsed and were set to NaT.")
        # Optionally drop these rows if dates are critical
        # df.dropna(subset=['posting_date'], inplace=True)
        # print(f"Dropped {failed_date_parses} rows with unparseable dates. New shape: {df.shape}")

    print(f"Step 5: Converted data types for year, odometer, posting_date.")

    # 6. Check Duplicates Again
    num_duplicates_after = df.duplicated().sum()
    if num_duplicates_after > 0:
        print(f"Step 6: Found {num_duplicates_after} duplicate rows after cleaning. Dropping them.")
        df.drop_duplicates(inplace=True)
        print(f"New shape after dropping duplicates: {df.shape}")
    else:
        print("Step 6: No duplicate rows found after cleaning.")

    # 7. Final Review
    print("\n--- Final Data Structure after Cleaning ---")
    df.info()
    print("\n--- Missing Values After Cleaning ---")
    print(df.isnull().sum()) # Should ideally show 0 for all columns now, except maybe posting_date if parsing failed

    # 8. Save Cleaned Data
    df.to_csv(output_file_path, index=False)
    print(f"\n--- Successfully saved cleaned data to: {output_file_path} ---")


except FileNotFoundError:
    print(f"Error: The file was not found at {input_file_path}")
    print("Please check the input directory and file name.")
except KeyError as e:
    print(f"Error: A specified column was not found: {e}. Check column names and cleaning steps.")
except Exception as e:
    print(f"An error occurred during the cleaning process: {e}")