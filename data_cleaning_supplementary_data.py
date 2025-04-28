import pandas as pd
import os
import re

# --- Configuration ---
# Directory containing the raw supplementary data files
input_supp_dir = "./supp_data"
# Directory where cleaned supplementary files will be saved
output_supp_dir = "./processed_supplementary_data" # Matches report

# Files to clean
files_to_clean = {
    "vehicles_epa.csv": "epa_cleaned.csv",
    "safecar_data.csv": "safety_cleaned.csv"
}

print(f"--- Starting Cleaning of Supplementary Data ---")
print(f"Input directory: {input_supp_dir}")
print(f"Output directory: {output_supp_dir}")

# --- Create output directory ---
try:
    os.makedirs(output_supp_dir, exist_ok=True)
    print(f"Ensured output directory exists: {output_supp_dir}")
except Exception as e:
    print(f"[ERROR] Creating output directory {output_supp_dir}: {e}")
    exit()

# --- Helper function to standardize column names ---
def standardize_col_names(df):
    """Converts column names to lowercase and replaces spaces/special chars with underscores."""
    cols = df.columns
    new_cols = []
    for col in cols:
        col = str(col).lower() # Ensure column name is string
        col = re.sub(r'\s+', '_', col) # Replace spaces with underscore
        col = re.sub(r'[^a-z0-9_]', '', col) # Remove non-alphanumeric chars except underscore
        new_cols.append(col)
    df.columns = new_cols
    return df

# --- Loop Through Files to Clean ---
for input_filename, output_filename in files_to_clean.items():
    print(f"\n\n{'='*20} Processing file: {input_filename} {'='*20}")
    input_path = os.path.join(input_supp_dir, input_filename)
    output_path = os.path.join(output_supp_dir, output_filename)

    # --- Load Data ---
    try:
        # Try reading with utf-8 first, fallback to latin1
        try:
            df = pd.read_csv(input_path, low_memory=False, encoding='utf-8')
        except UnicodeDecodeError:
            print(f"[INFO] UTF-8 failed. Trying 'latin1' encoding for {input_filename}.")
            df = pd.read_csv(input_path, low_memory=False, encoding='latin1')
        print(f"Successfully read: {input_filename}. Initial shape: {df.shape}")

    except FileNotFoundError:
        print(f"[ERROR] File not found at {input_path}. Skipping.")
        continue
    except Exception as e:
        print(f"[ERROR] Could not load data from {input_filename}: {e}. Skipping.")
        continue

    # Standardize column names first
    df = standardize_col_names(df)
    print(f"Standardized column names.")

    # --- Apply Cleaning Steps Based on Filename ---

    if input_filename == "vehicles_epa.csv":
        print("Applying cleaning steps for EPA data...")
        # 1. Select relevant columns
        cols_to_keep = [
            'year', 'make', 'model', 'vclass', # Identifiers
            'cylinders', 'displ', 'drive', 'trany', # Basic specs
            'fueltype1', # Primary fuel type
            'comb08', 'highway08', 'city08', # Standard MPG figures
            'co2tailpipegpm', # CO2 emissions
            'fuelcost08' # Annual fuel cost estimate
        ]
        # Ensure all desired columns actually exist after standardization
        cols_to_keep = [col for col in cols_to_keep if col in df.columns]
        print(f"Selecting columns: {cols_to_keep}")
        df = df[cols_to_keep]

        # 2. Rename columns for clarity
        rename_map = {
            'vclass': 'vehicle_class',
            'displ': 'displacement',
            'trany': 'transmission',
            'fueltype1': 'fuel_type',
            'comb08': 'combined_mpg',
            'highway08': 'highway_mpg',
            'city08': 'city_mpg',
            'co2tailpipegpm': 'co2_gpm',
            'fuelcost08': 'annual_fuel_cost'
        }
        # Apply renaming only for columns that exist
        rename_map = {k: v for k, v in rename_map.items() if k in df.columns}
        df.rename(columns=rename_map, inplace=True)
        print(f"Renamed columns: {list(rename_map.values())}")

        # 3. Handle Missing Values (Drop rows missing in selected columns)
        initial_rows = len(df)
        cols_to_check_na = df.columns # Check NAs in all selected columns
        df.dropna(subset=cols_to_check_na, inplace=True)
        rows_dropped = initial_rows - len(df)
        print(f"Dropped {rows_dropped} rows with missing values in selected columns.")

        # 4. Drop Duplicates (based on selected columns)
        initial_rows = len(df)
        df.drop_duplicates(inplace=True)
        rows_dropped = initial_rows - len(df)
        print(f"Dropped {rows_dropped} duplicate rows.")

    elif input_filename == "safecar_data.csv":
        print("Applying cleaning steps for Safety data...")

        # 1. Select relevant columns (Using standardized names from header)
        cols_to_keep = [
            'model_yr', 'make', 'model', # Identifiers (standardized)
            'overall_stars', # Primary rating (standardized)
            # Optionally add other specific ratings like:
            # 'frnt_driv_stars', 'frnt_pass_stars',
            # 'side_driv_stars', 'side_pass_stars',
            # 'side_pole_stars', 'rollover_stars'
        ]
        # Ensure all desired columns actually exist after standardization
        cols_to_keep_found = [col for col in cols_to_keep if col in df.columns]

        # Check if essential key columns were found after standardization
        essential_keys = ['model_yr', 'make', 'model']
        keys_found = all(key in df.columns for key in essential_keys)

        if not keys_found or not cols_to_keep_found:
             print(f"[WARNING] Could not find all expected key/rating columns ({essential_keys}, overall_stars) in safety data after standardization. Found: {cols_to_keep_found}. Skipping detailed cleaning.")
             continue # Skip further processing for this file

        print(f"Selecting columns: {cols_to_keep_found}")
        df = df[cols_to_keep_found]

        # 2. Rename columns (Using standardized source names)
        rename_map = {
            'model_yr': 'year',
            'overall_stars': 'safety_rating_overall'
            # Add other renames if you keep more rating columns
            # 'frnt_driv_stars': 'safety_rating_front_driver',
            # 'frnt_pass_stars': 'safety_rating_front_pass',
            # etc.
        }
        # Apply renaming only for columns that exist in the selected df
        rename_map = {k: v for k, v in rename_map.items() if k in df.columns}
        df.rename(columns=rename_map, inplace=True)
        print(f"Renamed columns: {list(rename_map.values())}")

        # 3. Handle Data Types for Safety Ratings (Often stored as text like '5' or 'Not Rated')
        # Identify rating columns based on *new* names if renamed, or original standardized names
        rating_cols_to_convert = [rename_map.get(col, col) for col in cols_to_keep_found if 'star' in col or 'rating' in col] # Find rating columns by name pattern
        print(f"Attempting numeric conversion for rating columns: {rating_cols_to_convert}")
        for col in rating_cols_to_convert:
             if col in df.columns:
                 # Attempt conversion to numeric, setting errors='coerce' turns non-numbers into NaN
                 df[col] = pd.to_numeric(df[col], errors='coerce')
                 print(f"Converted '{col}' to numeric (non-ratings become NaN).")
             else:
                 print(f"Column '{col}' not found for numeric conversion (likely not selected or already renamed).")


        # 4. Handle Missing Values (Drop rows missing keys or primary rating)
        initial_rows = len(df)
        # Define critical columns using the *final* names after renaming
        cols_to_check_na = ['year', 'make', 'model']
        if 'safety_rating_overall' in df.columns:
             cols_to_check_na.append('safety_rating_overall')
        # Ensure columns exist before checking NAs
        cols_to_check_na = [col for col in cols_to_check_na if col in df.columns]

        if cols_to_check_na: # Only drop if we have columns to check
             print(f"Checking for NAs in critical columns: {cols_to_check_na}")
             df.dropna(subset=cols_to_check_na, inplace=True)
             rows_dropped = initial_rows - len(df)
             print(f"Dropped {rows_dropped} rows with missing values in critical columns.")
        else:
             print("No critical columns found to check for NAs.")


        # 5. Drop Duplicates (based on selected columns)
        initial_rows = len(df)
        df.drop_duplicates(inplace=True)
        rows_dropped = initial_rows - len(df)
        print(f"Dropped {rows_dropped} duplicate rows.")

    else:
        print(f"No specific cleaning defined for {input_filename}. Skipping detailed cleaning.")
        continue # Skip saving if no cleaning applied

    # --- Save Cleaned Data ---
    try:
        df.to_csv(output_path, index=False)
        print(f"Successfully saved cleaned data to: {output_path}")
        print(f"Final shape: {df.shape}")
    except Exception as e:
        print(f"[ERROR] Could not save cleaned data for {input_filename}: {e}")


print(f"\n\n{'='*20} Cleaning of specified files complete {'='*20}")
