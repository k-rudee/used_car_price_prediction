import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import re # For standardizing merge keys and parsing titles
import glob # To find review files

# Scikit-learn imports
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer # To handle potential NaNs after merge
from scipy.stats import randint, uniform # For parameter distributions
from sklearn.inspection import PartialDependenceDisplay # For PDPs

# XGBoost import
from xgboost import XGBRegressor

# TensorFlow / Keras Imports
# Check if TensorFlow is available, otherwise skip NN part
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.callbacks import EarlyStopping
    TENSORFLOW_AVAILABLE = True
    # Set random seed for TensorFlow
    tf.random.set_seed(42)
except ImportError:
    print("[WARNING] TensorFlow not found. Neural Network model will be skipped. (Install with: pip install tensorflow)")
    TENSORFLOW_AVAILABLE = False

# Set random seed for reproducibility where possible
np.random.seed(42)
RANDOM_STATE = 42

# ==============================================================================
# 1. Load Data
# ==============================================================================
print("[INFO] Loading datasets...")

# --- Define file paths ---
# main_data_path = "./engineered_data/vehicles_engineered.csv"
# epa_data_path = "./processed_supplementary_data/epa_cleaned.csv"
# safety_data_path = "./processed_supplementary_data/safety_cleaned.csv"
# reliability_data_path = r"C:\Users\rudyk\OneDrive\Desktop\ML_Proj\supp_data\cleaned_jd_power_avg_reliability.csv" # Using absolute path
# review_data_dir = "./processed_reviews"
main_data_path = r"C:\Users\rudyk\OneDrive\Desktop\ML_Proj\engineered_data\vehicles_engineered.csv"
epa_data_path = r"C:\Users\rudyk\OneDrive\Desktop\ML_Proj\processed_supplementary_data\epa_cleaned.csv"
safety_data_path = r"C:\Users\rudyk\OneDrive\Desktop\ML_Proj\processed_supplementary_data\safety_cleaned.csv"
reliability_data_path = r"C:\Users\rudyk\OneDrive\Desktop\ML_Proj\supp_data\cleaned_jd_power_avg_reliability.csv"
review_data_dir = r"C:\Users\rudyk\OneDrive\Desktop\ML_Proj\processed_reviews"


# --- Load main vehicle data (Engineered) ---
try:
    try:
        df_main = pd.read_csv(main_data_path, parse_dates=['posting_date'])
    except (TypeError, ValueError):
         print("[INFO] Could not parse 'posting_date' automatically during load, attempting manual conversion later if needed.")
         df_main = pd.read_csv(main_data_path)
    print(f"  - Loaded main engineered vehicle data: {df_main.shape}")
    print(f"[VERIFY] Columns in main data: {df_main.columns.tolist()}")
except FileNotFoundError: exit(f"[ERROR] Main data file not found at {main_data_path}. Exiting.")
except Exception as e: exit(f"[ERROR] Failed to load main engineered data: {e}. Exiting.")

# --- Load supplementary data (handle potential errors) ---
def load_supplementary(path, name):
    try:
        df = pd.read_csv(path)
        print(f"  - Loaded {name} data: {df.shape}")
        return df
    except FileNotFoundError: print(f"[WARNING] {name} data file not found at {path}. Skipping merge."); return None
    except Exception as e: print(f"[WARNING] Failed to load {name} data from {path}: {e}. Skipping merge."); return None

df_epa = load_supplementary(epa_data_path, "EPA")
df_safety = load_supplementary(safety_data_path, "Safety")
df_reliability = load_supplementary(reliability_data_path, "Reliability")

# --- Load and Concatenate Review Data (Optional Step) ---
print("\n[INFO] Loading and concatenating review data...")
all_review_files = glob.glob(os.path.join(review_data_dir, 'cleaned_*.csv'))
if not all_review_files:
     print("[WARNING] No cleaned review files found in {review_data_dir}. Skipping review data processing.")
     df_reviews_all = None
else:
    review_dfs = []
    for f in all_review_files:
        try:
            try: df_temp = pd.read_csv(f, encoding='utf-8')
            except UnicodeDecodeError: df_temp = pd.read_csv(f, encoding='latin1')
            if all(col in df_temp.columns for col in ['author_name', 'vehicle_title', 'rating']): review_dfs.append(df_temp[['author_name', 'vehicle_title', 'rating']])
            else: print(f"[WARNING] Skipping review file {f} due to missing required columns.")
        except Exception as e: print(f"[WARNING] Failed to load or process review file {f}: {e}")
    if review_dfs:
        df_reviews_all = pd.concat(review_dfs, ignore_index=True)
        print(f"  - Concatenated {len(review_dfs)} review files. Total reviews: {df_reviews_all.shape}")
        df_reviews_all.dropna(subset=['author_name', 'vehicle_title', 'rating'], inplace=True)
        df_reviews_all.drop_duplicates(subset=['author_name', 'vehicle_title', 'rating'], inplace=True)
        df_reviews_all['rating'] = pd.to_numeric(df_reviews_all['rating'], errors='coerce')
        df_reviews_all.dropna(subset=['rating'], inplace=True)
        print(f"  - Shape after basic cleaning: {df_reviews_all.shape}")
    else: print("[WARNING] No valid review files loaded."); df_reviews_all = None

# ==============================================================================
# 2. Process Review Data (Optional Step)
# ==============================================================================
df_reviews_agg = None
if df_reviews_all is not None and not df_reviews_all.empty:
    print("\n[INFO] Processing review data...")
    title_regex = re.compile(r"(\d{4})\s+([a-zA-Z][a-zA-Z\s\-]*?)\s+([a-zA-Z0-9][a-zA-Z0-9\-\/\s]*?)(?:\s+\(|\s+LLC|\s+Ltd|\s+Inc|$)")
    def parse_vehicle_title(title):
        if pd.isna(title): return None, None, None
        match = title_regex.match(str(title))
        if match: return int(match.group(1)), match.group(2).strip(), ' '.join(match.group(3).strip().split()[:2])
        else: return None, None, None
    parsed_data = df_reviews_all['vehicle_title'].apply(parse_vehicle_title)
    df_reviews_all[['parsed_year', 'parsed_make', 'parsed_model']] = pd.DataFrame(parsed_data.tolist(), index=df_reviews_all.index)
    initial_rows = len(df_reviews_all); df_reviews_all.dropna(subset=['parsed_year', 'parsed_make', 'parsed_model'], inplace=True); rows_dropped = initial_rows - len(df_reviews_all)
    print(f"  - Dropped {rows_dropped} reviews where title parsing failed.")
    df_reviews_all['parsed_make'] = df_reviews_all['parsed_make'].str.lower().str.strip()
    df_reviews_all['parsed_model'] = df_reviews_all['parsed_model'].str.lower().str.strip()
    df_reviews_all['parsed_year'] = df_reviews_all['parsed_year'].astype(int)
    df_reviews_agg = df_reviews_all.groupby(['parsed_year', 'parsed_make', 'parsed_model']).agg(avg_review_rating=('rating', 'mean'), review_count=('rating', 'count')).reset_index()
    df_reviews_agg.rename(columns={'parsed_year': 'year', 'parsed_make': 'make', 'parsed_model': 'model'}, inplace=True)
    print(f"  - Aggregated reviews into {df_reviews_agg.shape[0]} unique Year/Make/Model combinations.")
else:
    print("\n[INFO] Skipping review data processing.")

# ==============================================================================
# 3. Merge Data
# ==============================================================================
print("\n[INFO] Merging datasets...")
df_merged = df_main.copy()
main_make_col = 'manufacturer' if 'manufacturer' in df_merged.columns else 'make'
if main_make_col not in df_merged.columns or 'model' not in df_merged.columns or 'year' not in df_merged.columns: exit("[ERROR] Main data missing key merge columns (year, make/manufacturer, model).")
print(f"  - Using '{main_make_col}' as primary make column.")
df_merged[main_make_col] = df_merged[main_make_col].astype(str).str.lower().str.strip()
df_merged['model'] = df_merged['model'].astype(str).str.lower().str.strip()
df_merged['year'] = pd.to_numeric(df_merged['year'], errors='coerce')
df_merged.dropna(subset=['year', main_make_col, 'model'], inplace=True); df_merged['year'] = df_merged['year'].astype(int)

common_merge_keys = ['year', 'make', 'model']
reliability_merge_key_left = main_make_col
reliability_merge_key_right = 'manufacturer'

# Function to perform merge and report
def merge_and_report(df_left, df_right, name, keys_left, keys_right=None, how='left'):
    if df_right is None: return df_left
    if keys_right is None: keys_right = keys_left
    if not all(k in df_right.columns for k in keys_right): print(f"[WARNING] {name} missing keys. Skipping."); return df_left
    # Basic standardization
    for i, k_right in enumerate(keys_right):
         k_left = keys_left[i]
         if k_left == 'year': df_right[k_right] = pd.to_numeric(df_right[k_right], errors='coerce').astype('Int64')
         else: df_right[k_right] = df_right[k_right].astype(str).str.lower().str.strip()
    df_right.dropna(subset=keys_right, inplace=True)
    df_right = df_right.drop_duplicates(subset=keys_right, keep='first')
    temp_make_added = False
    if 'make' in keys_left and 'make' not in df_left.columns and main_make_col == 'manufacturer': df_left['make'] = df_left['manufacturer']; temp_make_added = True
    df_merged_res = pd.merge(df_left, df_right, left_on=keys_left, right_on=keys_right, how=how, suffixes=('', f'_{name.lower()}'))
    if temp_make_added: df_merged_res.drop(columns=['make'], inplace=True)
    for i, k_right in enumerate(keys_right):
        k_left = keys_left[i]
        if k_right in df_merged_res.columns and k_left != k_right: df_merged_res.drop(columns=[k_right], inplace=True)
    print(f"  - Merged {name}. Shape: {df_merged_res.shape}")
    return df_merged_res

df_merged = merge_and_report(df_merged, df_epa, "EPA", common_merge_keys)
df_merged = merge_and_report(df_merged, df_safety, "Safety", common_merge_keys)
df_merged = merge_and_report(df_merged, df_reliability, "Reliability", [reliability_merge_key_left], [reliability_merge_key_right])
df_merged = merge_and_report(df_merged, df_reviews_agg, "Reviews", common_merge_keys)

# ==============================================================================
# 4. Feature Engineering
# ==============================================================================
print("\n[INFO] Performing feature engineering (checking if needed)...")
if 'vehicle_age' not in df_merged.columns:
    print("  - 'vehicle_age' not found, calculating...")
    if 'year' in df_merged.columns and 'posting_date' in df_merged.columns:
        df_merged['year'] = pd.to_numeric(df_merged['year'], errors='coerce')
        if not pd.api.types.is_datetime64_any_dtype(df_merged['posting_date']): df_merged['posting_date'] = pd.to_datetime(df_merged['posting_date'], errors='coerce', utc=True)
        df_merged.dropna(subset=['year', 'posting_date'], inplace=True); df_merged['year'] = df_merged['year'].astype(int)
        df_merged['vehicle_age'] = df_merged['posting_date'].dt.year - df_merged['year']; df_merged['vehicle_age'] = df_merged['vehicle_age'].apply(lambda x: max(0, x) if pd.notnull(x) else 0)
        print("  - Calculated 'vehicle_age'.")
    else: print("[ERROR] Cannot calculate vehicle_age: 'year' or 'posting_date' missing.")
else: print("  - 'vehicle_age' column already exists."); df_merged['vehicle_age'] = pd.to_numeric(df_merged['vehicle_age'], errors='coerce').fillna(0).astype(int); df_merged['vehicle_age'] = df_merged['vehicle_age'].apply(lambda x: max(0, x))
if 'log_price' not in df_merged.columns:
    print("  - 'log_price' not found, calculating...")
    if 'price' in df_merged.columns: df_merged['log_price'] = np.log1p(df_merged['price']); print("  - Created 'log_price'.")
    else: print("[ERROR] Cannot calculate log_price: 'price' column missing.")
else: print("  - 'log_price' column already exists."); df_merged['log_price'] = pd.to_numeric(df_merged['log_price'], errors='coerce')

# ==============================================================================
# 5. Preprocessing Setup
# ==============================================================================
print("\n[INFO] Setting up preprocessing...")
TARGET = 'log_price'
if TARGET not in df_merged.columns: exit(f"[ERROR] Target variable '{TARGET}' not found. Exiting.")
cols_to_exclude = [TARGET, 'price', 'posting_date', 'lat', 'long', 'region', 'model', 'make', 'manufacturer', 'price_per_mile', 'vehicle_title', 'author_name']
potential_features = [col for col in df_merged.columns if col not in cols_to_exclude and col != TARGET]
print(f"  - Initial potential features identified ({len(potential_features)}): {potential_features[:10]}...")
col_to_convert = 'cylinders'; new_col_name = col_to_convert + '_numeric'
if col_to_convert in df_merged.columns and col_to_convert in potential_features:
    print(f"\n[INFO] Processing column: '{col_to_convert}' for numerical conversion...")
    df_merged[new_col_name] = df_merged[col_to_convert].astype(str).str.extract(r'(\d+)').iloc[:, 0]
    df_merged[new_col_name] = pd.to_numeric(df_merged[new_col_name], errors='coerce')
    median_cyl = df_merged[new_col_name].median(); print(f"  - Calculated median for numeric cylinder values: {median_cyl}")
    nan_count_before = df_merged[new_col_name].isnull().sum(); df_merged[new_col_name].fillna(median_cyl, inplace=True); nan_count_after = df_merged[new_col_name].isnull().sum()
    print(f"  - Imputed {nan_count_before - nan_count_after} NaN values with median {median_cyl}.")
    if new_col_name not in potential_features: potential_features.append(new_col_name)
    if col_to_convert in potential_features: potential_features.remove(col_to_convert)
    print(f"  - Replaced categorical '{col_to_convert}' with numerical '{new_col_name}'.")
else: print(f"[INFO] Column '{col_to_convert}' not found or not in potential features. Skipping conversion.")
print("\n[INFO] Identifying final feature types...")
numerical_features = df_merged[potential_features].select_dtypes(include=np.number).columns.tolist()
categorical_features = df_merged[potential_features].select_dtypes(include=['object', 'category']).columns.tolist()
for col in ['year', 'vehicle_age']:
     if col in categorical_features: categorical_features.remove(col)
     if col in potential_features and col not in numerical_features: numerical_features.append(col)
if new_col_name in df_merged.columns and new_col_name not in numerical_features: numerical_features.append(new_col_name)
if new_col_name in categorical_features: categorical_features.remove(new_col_name)
print(f"[VERIFY] Final Numerical features ({len(numerical_features)}): {numerical_features}")
print(f"[VERIFY] Final Categorical features ({len(categorical_features)}): {categorical_features}")
numerical_pipeline = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
categorical_pipeline = Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='unknown')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
numerical_features = [col for col in numerical_features if col in df_merged[potential_features]]
categorical_features = [col for col in categorical_features if col in df_merged[potential_features]]
preprocessor = ColumnTransformer([('num', numerical_pipeline, numerical_features), ('cat', categorical_pipeline, categorical_features)], remainder='drop')
print("[INFO] Preprocessing pipelines defined.")

# ==============================================================================
# 6. Prepare Data for Modeling
# ==============================================================================
print("\n[INFO] Preparing data for modeling...")
final_feature_cols = numerical_features + categorical_features
final_feature_cols = [col for col in final_feature_cols if col in df_merged.columns]
print(f"[VERIFY] Final columns being used for X: {final_feature_cols}")
X = df_merged[final_feature_cols]
y = df_merged[TARGET]
if y.isnull().any(): print(f"[WARNING] Target '{TARGET}' has {y.isnull().sum()} NaNs. Dropping rows."); X = X[y.notna()]; y = y[y.notna()]
print(f"[VERIFY] Shape of X before split: {X.shape}"); print(f"[VERIFY] Shape of y before split: {y.shape}")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
print(f"  - Split data: Train shape {X_train.shape}, Test shape {X_test.shape}")
print("  - Fitting preprocessor and transforming data...")
X_train_processed = preprocessor.fit_transform(X_train); X_test_processed = preprocessor.transform(X_test)
print(f"[VERIFY] Shape of X_train_processed: {X_train_processed.shape}"); print(f"[VERIFY] Shape of X_test_processed: {X_test_processed.shape}")
try: feature_names_out = preprocessor.get_feature_names_out()
except Exception: num_processed_features = X_train_processed.shape[1]; feature_names_out = [f'feature_{i}' for i in range(num_processed_features)]
print(f"  - Final number of features after OHE: {len(feature_names_out)}")

# ==============================================================================
# 7. Define Metadata
# ==============================================================================
metadata = {
    'feature_names': list(feature_names_out), 'target_variable': TARGET, 'target_description': 'Log(Price + 1)',
    'feature_types': {'numerical_scaled': [n for n in feature_names_out if n.startswith('num__')], 'categorical_encoded': [n for n in feature_names_out if n.startswith('cat__')]},
    'data_source_description': 'Engineered Vehicle Listings merged with EPA, Safety, Reliability, and Aggregated Review Data'
}
print("\n[INFO] Final Metadata Defined:")
print(f"  - Target: {metadata['target_variable']} ({metadata['target_description']})")
print(f"  - Number of Features: {len(metadata['feature_names'])}")
print(f"  - Data Source: {metadata['data_source_description']}")

# ==============================================================================
# 8. Define Evaluation Function
# ==============================================================================
def evaluate_model(model, model_name, X_train, y_train, X_test, y_test, is_keras=False):
    """Evaluates a pre-trained model and returns metrics."""
    print(f"\n--- Evaluating {model_name} ---")
    print(f"[VERIFY] Evaluating with Test data shape: X={X_test.shape}, y={y_test.shape}")
    start_time = time.time()
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    eval_time = time.time() - start_time
    print(f"Prediction finished in {eval_time:.2f} seconds.")
    if is_keras: y_pred_train = y_pred_train.flatten(); y_pred_test = y_pred_test.flatten()
    train_r2 = r2_score(y_train, y_pred_train); test_r2 = r2_score(y_test, y_pred_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    results = {'model_name': model_name, 'train_r2': train_r2, 'test_r2': test_r2, 'test_rmse': test_rmse, 'y_pred_train': y_pred_train, 'y_pred_test': y_pred_test, 'model_object': model}
    print("\n" + "="*30); print(f"  Model Performance: {model_name}"); print("="*30)
    print(f"  Training R² Score : {train_r2:.4f}"); print(f"  Test R² Score     : {test_r2:.4f}"); print(f"  Test RMSE         : {test_rmse:.4f}"); print("="*30 + "\n")
    return results

# ==============================================================================
# 9. Model Training (Including Neural Network)
# ==============================================================================
all_results = []
X_train_final = X_train_processed
X_test_final = X_test_processed

# --- Model 1: Linear Regression (Baseline) ---
print("\n--- Training Linear Regression (Baseline) ---")
lr_model = LinearRegression()
lr_model.fit(X_train_final, y_train)
lr_results = evaluate_model(lr_model, "Linear Regression", X_train_final, y_train, X_test_final, y_test)
all_results.append(lr_results)

# --- Model 2: Random Forest Tuning (Optimized) ---
print("\n--- Tuning Random Forest with RandomizedSearchCV (Optimized) ---")
param_dist_rf = {'n_estimators': randint(100, 400), 'max_depth': [10, 20, 30, None], 'min_samples_split': randint(10, 30), 'min_samples_leaf': randint(5, 15), 'max_features': [0.6, 0.8, 'sqrt']}
rf_base = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)
rf_search = RandomizedSearchCV(estimator=rf_base, param_distributions=param_dist_rf, n_iter=10, cv=3, scoring='r2', n_jobs=-1, random_state=RANDOM_STATE, verbose=1)
start_time = time.time(); rf_search.fit(X_train_final, y_train); end_time = time.time()
print(f"Random Forest tuning completed in {end_time - start_time:.2f} seconds.")
print(f"[REPORT] Best Parameters found for Random Forest: {rf_search.best_params_}")
print(f"[REPORT] Best CV R² Score for Random Forest: {rf_search.best_score_:.4f}")
best_rf_model = rf_search.best_estimator_
rf_results = evaluate_model(best_rf_model, "Random Forest (Tuned)", X_train_final, y_train, X_test_final, y_test)
all_results.append(rf_results)

# --- Model 3: XGBoost Tuning (Optimized) ---
print("\n--- Tuning XGBoost with RandomizedSearchCV (Optimized) ---")
param_dist_xgb = {'n_estimators': randint(100, 500), 'learning_rate': uniform(0.02, 0.15), 'max_depth': randint(5, 10), 'subsample': uniform(0.7, 0.3), 'colsample_bytree': uniform(0.7, 0.3), 'gamma': uniform(0, 0.3), 'reg_alpha': uniform(0, 0.5), 'reg_lambda': uniform(1, 3)}
xgb_base = XGBRegressor(random_state=RANDOM_STATE, n_jobs=-1, eval_metric='rmse')
xgb_search = RandomizedSearchCV(estimator=xgb_base, param_distributions=param_dist_xgb, n_iter=15, cv=3, scoring='r2', n_jobs=-1, random_state=RANDOM_STATE, verbose=1)
start_time = time.time(); xgb_search.fit(X_train_final, y_train); end_time = time.time()
print(f"XGBoost tuning completed in {end_time - start_time:.2f} seconds.")
print(f"[REPORT] Best Parameters found for XGBoost: {xgb_search.best_params_}")
print(f"[REPORT] Best CV R² Score for XGBoost: {xgb_search.best_score_:.4f}")
best_xgb_model = xgb_search.best_estimator_
xgb_results = evaluate_model(best_xgb_model, "XGBoost (Tuned)", X_train_final, y_train, X_test_final, y_test)
all_results.append(xgb_results)

# --- Model 4: Neural Network (Keras) ---
if TENSORFLOW_AVAILABLE:
    print("\n--- Defining and Training Neural Network (Keras) ---")
    def build_nn_model(input_shape):
        """Builds a simple sequential Keras model."""
        model = keras.Sequential([
            layers.Input(shape=(input_shape,), name="Input"),
            layers.Dense(128, activation='relu', name="Dense_1"), layers.Dropout(0.3, name="Dropout_1"),
            layers.Dense(64, activation='relu', name="Dense_2"), layers.Dropout(0.3, name="Dropout_2"),
            layers.Dense(32, activation='relu', name="Dense_3"),
            layers.Dense(1, activation='linear', name="Output")
        ], name="FeedForward_NN")
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mae', 'mse'])
        return model

    input_shape = X_train_final.shape[1]
    nn_model = build_nn_model(input_shape)
    print("\n[REPORT] Neural Network Model Summary:")
    nn_model.summary(print_fn=lambda x: print(f"  {x}"))

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)

    print("\n[INFO] Training Neural Network (Epoch progress will be shown)...")
    start_time = time.time()
    history = nn_model.fit(
        X_train_final, y_train, epochs=100, batch_size=128,
        validation_split=0.2, callbacks=[early_stopping], verbose=1
    )
    end_time = time.time()
    print(f"Neural Network training completed in {end_time - start_time:.2f} seconds.")
    final_epoch = len(history.history['loss'])
    print(f"[REPORT] Neural Network trained for {final_epoch} epochs (Early Stopping Triggered: {'Yes' if final_epoch < 100 else 'No'}).")

    nn_results = evaluate_model(nn_model, "Neural Network", X_train_final, y_train, X_test_final, y_test, is_keras=True)
    all_results.append(nn_results)
else:
    print("\n[INFO] Skipping Neural Network training as TensorFlow is not available.")


# ==============================================================================
# 10. Final Summary Table
# ==============================================================================
print("\n" + "#"*65) # Adjusted width
print("                     Model Performance Summary") # Centered text
print("#"*65) # Adjusted width
print(f"{'Model':<25} | {'Training R²':<15} | {'Test R²':<15} | {'Test RMSE':<15}") # Adjusted padding
print("-"*65) # Adjusted width
all_results_sorted = sorted(all_results, key=lambda x: x['test_r2'], reverse=True)
for result in all_results_sorted:
    print(f"{result['model_name']:<25} | {result['train_r2']:<15.4f} | {result['test_r2']:<15.4f} | {result['test_rmse']:<15.4f}")
print("#"*65 + "\n") # Adjusted width


# ==============================================================================
# 11. Visualizations
# ==============================================================================
print("[INFO] Generating visualizations...")
output_viz_dir = "./model_visualizations" # Save plots to a new directory
os.makedirs(output_viz_dir, exist_ok=True)

# --- 11a. Actual vs. Predicted Plots ---
def plot_actual_vs_predicted(y_true, y_pred, model_name, target_desc, save_dir):
    """Generates an actual vs. predicted scatter plot."""
    plt.figure(figsize=(8, 8))
    sample_size = min(len(y_true), 50000); indices = np.random.choice(len(y_true), sample_size, replace=False)
    y_true_sampled = y_true.iloc[indices] if isinstance(y_true, pd.Series) else y_true[indices]
    y_pred_sampled = y_pred[indices]
    plt.scatter(y_true_sampled, y_pred_sampled, alpha=0.2, edgecolors='none', s=15)
    min_val = min(np.min(y_true), np.min(y_pred)) * 0.95; max_val = max(np.max(y_true), np.max(y_pred)) * 1.05
    plt.plot([min_val, max_val], [min_val, max_val], '--', color='red', lw=2, label='Ideal Fit (y=x)')
    plt.xlim(min_val, max_val); plt.ylim(min_val, max_val)
    plt.xlabel(f"Actual {target_desc}", fontsize=12); plt.ylabel(f"Predicted {target_desc}", fontsize=12)
    plt.title(f"Actual vs. Predicted Values ({model_name})", fontsize=14); plt.legend(); plt.grid(True); plt.tight_layout()
    safe_model_name = re.sub(r'[^\w\-]+', '_', model_name.lower()); filename = os.path.join(save_dir, f"actual_vs_predicted_{safe_model_name}.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight'); print(f"  - Saved Actual vs Predicted plot for {model_name} to {filename}"); plt.close()

for result in all_results: plot_actual_vs_predicted(y_test, result['y_pred_test'], result['model_name'], metadata['target_description'], output_viz_dir)

# --- 11b. Residual Plots ---
def plot_residuals(y_true, y_pred, model_name, target_desc, save_dir):
    """Generates a predicted vs. residuals scatter plot."""
    residuals = y_true - y_pred
    plt.figure(figsize=(8, 6))
    sample_size = min(len(y_true), 50000) # Sample for performance
    indices = np.random.choice(len(y_true), sample_size, replace=False)
    residuals_sampled = residuals.iloc[indices] if isinstance(residuals, pd.Series) else residuals[indices]
    y_pred_sampled = y_pred[indices]
    plt.scatter(y_pred_sampled, residuals_sampled, alpha=0.2, edgecolors='none', s=15)
    plt.axhline(y=0, color='red', linestyle='--', lw=2)
    plt.xlabel(f"Predicted {target_desc}", fontsize=12)
    plt.ylabel("Residuals (Actual - Predicted)", fontsize=12)
    plt.title(f"Residual Plot ({model_name})", fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    safe_model_name = re.sub(r'[^\w\-]+', '_', model_name.lower()); filename = os.path.join(save_dir, f"residual_plot_{safe_model_name}.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight'); print(f"  - Saved Residual plot for {model_name} to {filename}"); plt.close()

for result in all_results: plot_residuals(y_test, result['y_pred_test'], result['model_name'], metadata['target_description'], output_viz_dir)


# --- 11c. Feature Importance Plots ---
def plot_feature_importance(model, feature_names, model_name, save_dir, top_n=25):
    """Generates feature importance plot and returns top feature names."""
    top_feature_names = [] # Initialize list to return
    if not feature_names: print(f"[WARNING] No feature names for {model_name}. Skipping plot."); return top_feature_names
    if hasattr(model, 'feature_importances_'): importances = model.feature_importances_
    elif hasattr(model, 'coef_'): importances = np.abs(model.coef_); importances = importances.flatten()
    else: print(f"  - Feature importances not available for {model_name}."); return top_feature_names
    if len(importances) != len(feature_names): print(f"[WARNING] Mismatch counts for {model_name}. Skipping plot."); return top_feature_names
    simple_feature_names = [re.sub(r'^(num__|cat__|remainder__)', '', fn) for fn in feature_names]
    feature_importance_df = pd.DataFrame({'feature': simple_feature_names, 'importance': importances}).sort_values('importance', ascending=False).head(top_n)
    top_feature_names = feature_importance_df['feature'].tolist()
    plt.figure(figsize=(10, max(6, len(feature_importance_df) * 0.4)))
    plt.barh(feature_importance_df['feature'], feature_importance_df['importance'], align='center')
    plt.xlabel('Feature Importance Score / |Coefficient|', fontsize=12); plt.ylabel('Feature', fontsize=12)
    plt.title(f'Top {len(feature_importance_df)} Feature Importances ({model_name})', fontsize=14)
    plt.gca().invert_yaxis(); plt.tight_layout()
    safe_model_name = re.sub(r'[^\w\-]+', '_', model_name.lower()); filename = os.path.join(save_dir, f"feature_importance_{safe_model_name}.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight'); print(f"  - Saved Feature Importance plot for {model_name} to {filename}"); plt.close()
    return top_feature_names

# Plot for RF and XGBoost using final feature names
top_rf_features = []
top_xgb_features = []
final_feature_names = metadata.get('feature_names', [])
if final_feature_names:
    top_rf_features = plot_feature_importance(rf_results['model_object'], final_feature_names, rf_results['model_name'], output_viz_dir)
    top_xgb_features = plot_feature_importance(xgb_results['model_object'], final_feature_names, xgb_results['model_name'], output_viz_dir)
else: print("[WARNING] Final feature names not found. Skipping feature importance plots.")


# --- 11d. Partial Dependence Plots ---
# Choose the best model (e.g., based on test R2)
if all_results: # Check if results list is not empty
    # Exclude NN from being chosen for PDP in this setup
    valid_models_for_pdp = [r for r in all_results if isinstance(r['model_object'], (LinearRegression, RandomForestRegressor, XGBRegressor))]
    if valid_models_for_pdp:
        best_model_result = sorted(valid_models_for_pdp, key=lambda x: x['test_r2'], reverse=True)[0]
        best_model = best_model_result['model_object']
        best_model_name = best_model_result['model_name']

        print(f"\n[INFO] Generating Partial Dependence Plots for the best non-NN model: {best_model_name}")
        top_features_for_pdp_simple = list(set(top_rf_features + top_xgb_features))[:6] # Example: Top 6 unique features from both models
        print(f"  - Selected top features for PDP: {top_features_for_pdp_simple}")
        pdp_feature_indices = []; pdp_feature_names_original = []
        for simple_name in top_features_for_pdp_simple:
             # Find the original numerical feature name before scaling/imputation
             original_name_match = [orig for orig in numerical_features if orig == simple_name]
             if original_name_match:
                 original_name = original_name_match[0]
                 pdp_feature_names_original.append(original_name)
                 try: idx = X_train.columns.get_loc(original_name); pdp_feature_indices.append(idx)
                 except KeyError: print(f"  - Could not find index for PDP feature: {original_name}")
             # else: # Optionally handle categorical features if needed, though less common for PDP
             #     print(f"  - Skipping non-numerical feature for PDP: {simple_name}")

        print(f"  - Plotting PDP for original features: {pdp_feature_names_original}")
        if pdp_feature_names_original:
            try:
                n_pdp_samples = min(len(X_train), 1000); X_train_sample = X_train.sample(n=n_pdp_samples, random_state=RANDOM_STATE)
                full_pipeline = Pipeline([('preprocessor', preprocessor), ('regressor', best_model)])
                print(f"  - Calculating PDP for {len(pdp_feature_names_original)} features using {n_pdp_samples} samples...")
                start_pdp_time = time.time()
                fig, ax = plt.subplots(figsize=(max(12, len(pdp_feature_names_original)*3), 4), ncols=len(pdp_feature_names_original), constrained_layout=True)
                if len(pdp_feature_names_original) == 1: ax = [ax] # Handle single subplot case
                display = PartialDependenceDisplay.from_estimator(full_pipeline, X_train_sample, features=pdp_feature_names_original, ax=ax, n_jobs=-1, grid_resolution=50)
                fig.suptitle(f'Partial Dependence Plots ({best_model_name})', fontsize=16, y=1.05)
                end_pdp_time = time.time(); print(f"  - PDP calculation finished in {end_pdp_time - start_pdp_time:.2f} seconds.")
                safe_model_name = re.sub(r'[^\w\-]+', '_', best_model_name.lower()); filename = os.path.join(output_viz_dir, f"pdp_plot_{safe_model_name}.png")
                plt.savefig(filename, dpi=300, bbox_inches='tight'); print(f"  - Saved PDP plot to {filename}"); plt.close(fig)
            except Exception as e: print(f"[ERROR] Failed to generate Partial Dependence Plots: {e}")
        else: print("  - No suitable top numerical features found for PDP plotting.")
    else:
         print("[INFO] No suitable (non-NN) models found for PDP plots.")
else:
     print("[INFO] No models evaluated, skipping PDP plots.")

# --- 11e. Neural Network Training History Plot (NEW) ---
def plot_nn_history(history, model_name, save_dir):
    """Plots training & validation loss and metrics for a Keras model."""
    if history is None or not hasattr(history, 'history') or not history.history:
        print(f"[WARNING] No training history found or history is empty for {model_name}. Skipping history plot.")
        return

    history_df = pd.DataFrame(history.history)
    if 'loss' not in history_df.columns or 'val_loss' not in history_df.columns:
         print(f"[WARNING] 'loss' or 'val_loss' not found in history for {model_name}. Skipping history plot.")
         return

    plt.figure(figsize=(12, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(history_df['loss'], label='Training Loss')
    plt.plot(history_df['val_loss'], label='Validation Loss')
    plt.title(f'{model_name} - Loss Over Epochs', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (MSE)', fontsize=12)
    plt.legend()
    plt.grid(True)

    # Plot MAE (if available)
    if 'mae' in history_df.columns and 'val_mae' in history_df.columns:
        plt.subplot(1, 2, 2)
        plt.plot(history_df['mae'], label='Training MAE')
        plt.plot(history_df['val_mae'], label='Validation MAE')
        plt.title(f'{model_name} - MAE Over Epochs', fontsize=14)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Mean Absolute Error', fontsize=12)
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    safe_model_name = re.sub(r'[^\w\-]+', '_', model_name.lower()); filename = os.path.join(save_dir, f"nn_training_history_{safe_model_name}.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight'); print(f"  - Saved NN Training History plot for {model_name} to {filename}"); plt.close()

# Plot history only if NN was trained and results exist
if TENSORFLOW_AVAILABLE and 'nn_results' in locals():
    plot_nn_history(history, nn_results['model_name'], output_viz_dir)


print("\n[INFO] Visualization generation complete.")
print("[INFO] Script finished.")

