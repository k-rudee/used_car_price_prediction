import pandas as pd
import numpy as np
import os
import warnings

# --- Configuration ---
# Directory where the cleaned input file is located
cleaned_data_dir = "./main_data"
# Cleaned file name
cleaned_file_name = "vehicles_cleaned.csv"
# Directory where the engineered file will be saved
output_dir = "./engineered_data"
# Output file name
output_file_name = "vehicles_engineered.csv"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Construct full file paths
input_file_path = os.path.join(cleaned_data_dir, cleaned_file_name)
output_file_path = os.path.join(output_dir, output_file_name)

print(f"--- Starting Feature Engineering ---")
print(f"Input file: {input_file_path}")
print(f"Output file will be: {output_file_path}")

# --- Data Loading ---
try:
    # Load data, keep posting_date as object/string unless needed as datetime elsewhere
    df = pd.read_csv(input_file_path)
    print(f"\n--- Successfully loaded cleaned data. Shape: {df.shape} ---")
    print(f"Initial columns: {df.columns.tolist()}")

    # --- Feature Engineering Steps ---

    # 1. Price per Mile
    if 'price' in df.columns and 'odometer' in df.columns:
         # Add 1 to odometer to prevent division by zero or issues with very low values
        with warnings.catch_warnings(): # Suppress potential warning from dividing by near-zero odometer
            warnings.simplefilter("ignore", category=RuntimeWarning)
            # Use .loc to avoid SettingWithCopyWarning
            df.loc[:, 'price_per_mile'] = df['price'] / (df['odometer'] + 1)
        print(f"Step 1: Calculated 'price_per_mile'. NaNs: {df['price_per_mile'].isnull().sum()}")
    else:
        print("Step 1: Skipped 'price_per_mile' calculation - 'price' or 'odometer' column missing.")

    # 2. Log Transformation of Price (Target Variable)
    if 'price' in df.columns:
        # Use .loc to avoid SettingWithCopyWarning
        df.loc[:, 'log_price'] = np.log1p(df['price'])
        print(f"Step 2: Applied log transformation to 'price', creating 'log_price'.")
    else:
        print("Step 2: Skipped 'log_price' calculation - 'price' column missing.")

    # 3. Simplified Categorical Features (Example: High Cardinality Handling)
    # Reduce cardinality of 'manufacturer' and 'model' by grouping rare ones
    for col in ['manufacturer', 'model']:
        if col in df.columns:
            threshold = 50 # Keep categories appearing at least 50 times
            counts = df[col].value_counts()
            rare_cats = counts[counts < threshold].index.tolist()
            if rare_cats:
                print(f"Step 3: Grouping {len(rare_cats)} rare categories in '{col}' into 'Other_{col}'.")
                # Use .loc to modify DataFrame in place safely
                df.loc[df[col].isin(rare_cats), col] = f'Other_{col}'
            else:
                 print(f"Step 3: No rare categories found below threshold {threshold} for '{col}'.")

    # 4. Drop Original Price Column
    cols_to_drop = ['price'] # Only drop original price, keep log_price. Keep posting_date.
    # Only drop columns that actually exist
    cols_to_drop = [col for col in cols_to_drop if col in df.columns]
    if cols_to_drop:
        df.drop(columns=cols_to_drop, inplace=True)
        print(f"Step 4: Dropped original column: {cols_to_drop}")

    # --- Final Review ---
    print("\n--- Data Structure after Feature Engineering ---")
    df.info()
    print("\n--- Missing Values After Engineering ---")
    print(df.isnull().sum())
    print("\n--- First 5 Rows ---")
    print(df.head())

    # --- Save Engineered Data ---
    df.to_csv(output_file_path, index=False)
    print(f"\n--- Successfully saved engineered data to: {output_file_path} ---")


except FileNotFoundError:
    print(f"Error: The file was not found at {input_file_path}")
    print("Please check the input directory and file name.")
except KeyError as e:
    print(f"Error: A specified column was not found during feature engineering: {e}.")
except Exception as e:
    print(f"An unexpected error occurred during feature engineering: {e}")