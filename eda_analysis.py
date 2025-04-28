import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Configuration ---
input_dir = "./cleaned_data"
input_file = "vehicles_cleaned.csv"
output_viz_dir = "./visualizations" # Directory to save plots

file_path = os.path.join(input_dir, input_file)

# Create visualization output directory if it doesn't exist
os.makedirs(output_viz_dir, exist_ok=True)

# Set plot style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("viridis") # Set a color palette

print(f"--- Starting EDA for {file_path} ---")

# --- Load Data ---
try:
    df = pd.read_csv(file_path, parse_dates=['posting_date'])
    print(f"\n[INFO] Successfully loaded data. Shape: {df.shape}")
except FileNotFoundError:
    print(f"[ERROR] File not found at {file_path}. Please ensure the path is correct.")
    exit()
except Exception as e:
    print(f"[ERROR] Could not load data: {e}")
    exit()

# --- 1. Target Variable Analysis (Price) ---
print("\n--- 1. Target Variable Analysis (Price) ---")
price_skew = df['price'].skew()
print(f"[REPORT] Distribution of Price:")
print(f" - Skewness: {price_skew:.2f} (Indicates significant positive skew)")
print(f" - Mean Price: ${df['price'].mean():,.2f}")
print(f" - Median Price: ${df['price'].median():,.2f}")
print(f" - Min Price: ${df['price'].min():,.0f}, Max Price: ${df['price'].max():,.0f}")

# Apply log transformation (log(x+1) to handle potential zeros if filtering was different)
df['log_price'] = np.log1p(df['price'])
log_price_skew = df['log_price'].skew()
print(f"\n[REPORT] Distribution of Log-Transformed Price (log_price):")
print(f" - Skewness after log transform: {log_price_skew:.2f} (Much more symmetric)")
print(f" - Mean Log Price: {df['log_price'].mean():.2f}")
print(f" - Median Log Price: {df['log_price'].median():.2f}")


# --- Visualization 1: Distribution of Log Price ---
plt.figure(figsize=(10, 6))
sns.histplot(df['log_price'], kde=True, bins=50)
plt.title('Distribution of Used Car Prices (Log Scale)', fontsize=16)
plt.xlabel('Log(Price + 1)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.grid(axis='y', alpha=0.75)
viz1_path = os.path.join(output_viz_dir, 'price_distribution_log.png')
plt.savefig(viz1_path, dpi=300, bbox_inches='tight')
print(f"\n[PLOT] Saved Log Price distribution plot to {viz1_path}")
# plt.show() # Uncomment to display plot if running interactively


# --- 2. Key Numerical Predictors (Year, Odometer) ---
print("\n--- 2. Key Numerical Predictors ---")
print("[REPORT] Summary statistics for Year:")
print(df['year'].describe().to_string())
print("\n[REPORT] Summary statistics for Odometer:")
print(df['odometer'].describe().to_string())

# Calculate Correlations with log_price
corr_year = df['log_price'].corr(df['year'])
corr_odo = df['log_price'].corr(df['odometer'])
print(f"\n[REPORT] Correlation with Log Price:")
print(f" - Year vs log_price: {corr_year:.2f}")
print(f" - Odometer vs log_price: {corr_odo:.2f}")


# --- Visualization 2: Price vs Year and Odometer ---
fig, axes = plt.subplots(1, 2, figsize=(16, 6)) # 1 row, 2 columns

# Plot 2a: Price vs Year (Hexbin)
hb_year = axes[0].hexbin(df['year'], df['log_price'], gridsize=50, cmap='viridis', mincnt=1)
axes[0].set_title('Log Price vs. Vehicle Year', fontsize=14)
axes[0].set_xlabel('Year', fontsize=12)
axes[0].set_ylabel('Log(Price + 1)', fontsize=12)
cb_year = fig.colorbar(hb_year, ax=axes[0])
cb_year.set_label('Counts')

# Plot 2b: Price vs Odometer (Hexbin)
# Use log scale for odometer for better visualization if needed, but linear might be clearer
hb_odo = axes[1].hexbin(df['odometer'], df['log_price'], gridsize=50, cmap='viridis', mincnt=1) # Using linear scale for odometer
axes[1].set_title('Log Price vs. Odometer Reading', fontsize=14)
axes[1].set_xlabel('Odometer (miles)', fontsize=12)
axes[1].set_ylabel('Log(Price + 1)', fontsize=12)
# Format odometer axis
axes[1].ticklabel_format(style='plain', axis='x')
cb_odo = fig.colorbar(hb_odo, ax=axes[1])
cb_odo.set_label('Counts')

plt.tight_layout()
viz2_path = os.path.join(output_viz_dir, 'price_vs_year_odometer_hex.png')
plt.savefig(viz2_path, dpi=300, bbox_inches='tight')
print(f"\n[PLOT] Saved Price vs Year/Odometer plots to {viz2_path}")
# plt.show() # Uncomment to display plot if running interactively


# --- 3. Key Categorical Predictors ---
print("\n--- 3. Key Categorical Predictors ---")
categorical_cols = ['manufacturer', 'condition', 'cylinders', 'fuel', 'transmission', 'drive', 'type', 'paint_color', 'title_status']

# Calculate percentage of 'unknown' for relevant columns
print("[REPORT] Percentage of 'unknown' values in key imputed columns:")
for col in ['condition', 'cylinders', 'drive', 'type', 'paint_color']:
    if col in df.columns:
        unknown_perc = (df[col] == 'unknown').mean() * 100
        print(f" - {col}: {unknown_perc:.1f}%")

# Top Manufacturers
print("\n[REPORT] Top 10 Manufacturers by Frequency:")
top_manufacturers = df['manufacturer'].value_counts().head(10)
print(top_manufacturers.to_string())

# Median price by condition
print("\n[REPORT] Median Log Price by Condition:")
median_price_by_cond = df.groupby('condition')['log_price'].median().sort_values(ascending=False)
print(median_price_by_cond.to_string())


# --- Visualization 3: Price vs Top Manufacturers ---
# Get top N manufacturers for visualization
n_top = 15
top_n_manufacturers = df['manufacturer'].value_counts().nlargest(n_top).index.tolist()
df_top_mfg = df[df['manufacturer'].isin(top_n_manufacturers)]

# Calculate median prices for ordering the plot
median_prices = df_top_mfg.groupby('manufacturer')['log_price'].median().sort_values(ascending=False)
order = median_prices.index # Order boxes by median price

plt.figure(figsize=(14, 8))
sns.boxplot(data=df_top_mfg, x='manufacturer', y='log_price', order=order, palette='viridis', showfliers=False) # Hide outliers for clarity
plt.title(f'Log Price Distribution by Top {n_top} Manufacturers', fontsize=16)
plt.xlabel('Manufacturer', fontsize=12)
plt.ylabel('Log(Price + 1)', fontsize=12)
plt.xticks(rotation=45, ha='right') # Rotate labels for readability
plt.grid(axis='y', alpha=0.75)
plt.tight_layout()
viz3_path = os.path.join(output_viz_dir, 'price_vs_top_manufacturers.png')
plt.savefig(viz3_path, dpi=300, bbox_inches='tight')
print(f"\n[PLOT] Saved Price vs Top Manufacturers plot to {viz3_path}")
# plt.show() # Uncomment to display plot if running interactively


# --- 4. Correlation Analysis ---
print("\n--- 4. Correlation Analysis ---")
# Select numerical columns for correlation heatmap
corr_cols = ['log_price', 'year', 'odometer']
# Check if 'cylinders' was successfully converted to numeric (if intended in cleaning - recall it was left as object/'unknown')
# If cylinders is numeric add it: if pd.api.types.is_numeric_dtype(df['cylinders']): corr_cols.append('cylinders')
# If lat/long are relevant: corr_cols.extend(['lat', 'long'])

correlation_matrix = df[corr_cols].corr()
print("[REPORT] Correlation Matrix for Key Numerical Features:")
print(correlation_matrix.to_string())

# Optional: Plot heatmap
# plt.figure(figsize=(8, 6))
# sns.heatmap(correlation_matrix, annot=True, cmap='viridis', fmt=".2f")
# plt.title('Correlation Heatmap of Key Numerical Features')
# plt.tight_layout()
# heatmap_path = os.path.join(output_viz_dir, 'correlation_heatmap.png')
# plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
# print(f"\n[PLOT] Saved Correlation Heatmap to {heatmap_path}")
# plt.show()


# --- 5. Time Aspect ---
print("\n--- 5. Time Aspect (Posting Date) ---")
min_date = df['posting_date'].min()
max_date = df['posting_date'].max()
date_range_days = (max_date - min_date).days
print(f"[REPORT] Posting Date Range:")
print(f" - Earliest Posting: {min_date.strftime('%Y-%m-%d')}")
print(f" - Latest Posting: {max_date.strftime('%Y-%m-%d')}")
print(f" - Data spans approximately {date_range_days} days.")
# Further time series analysis would require merging with external data or longer dataset span.


print("\n--- EDA Complete ---")