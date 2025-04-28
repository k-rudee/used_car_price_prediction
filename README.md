# Used Car Price Prediction with Machine Learning 🚗💰

This repository contains the code and analysis for the ISYE 6740 final project, focusing on exploring the U.S. used car market and predicting vehicle prices using machine learning techniques.

## Project Overview

The used car market is complex, with prices influenced by numerous factors like mileage, age, brand, condition, and specifications. This project aims to:

1.  **Analyze Market Dynamics:** Conduct Exploratory Data Analysis (EDA) to understand key trends and relationships between vehicle attributes and prices in the U.S. used car market.
2.  **Predict Prices:** Develop, train, and compare various supervised machine learning models (Linear Regression, Random Forest, XGBoost, Neural Network) to accurately predict the final sale prices of used vehicles.
3.  **Integrate Diverse Data:** Utilize a primary dataset of used car listings and enrich it with external data sources like EPA fuel economy ratings, NHTSA safety ratings, and user reviews for a more comprehensive valuation model.

## Data Sources

*   **Primary:** Craigslist Cars + Trucks Data (Kaggle) - ~426k U.S. used car listings (April-May 2021) with details like make, model, year, mileage, condition, price.
*   **Supplementary:**
    *   EPA Fuel Economy Data (FuelEconomy.gov) - MPG ratings, fuel costs, CO2 emissions.
    *   NHTSA Vehicle Safety Ratings (NHTSA) - Overall safety star ratings.
    *   Edmunds Consumer Car Ratings and Reviews (Kaggle) - User reviews and ratings.
    *   *(Note: JD Power reliability and macroeconomic data were explored but faced integration challenges due to data limitations).*

## Repository Structure & File Descriptions

This repository is organized with Python scripts handling different stages of the project:

*   **`data_cleaning_vehicle_listing_data.py`**: Cleans the primary Kaggle used car listings dataset. Handles missing values, removes irrelevant columns, filters outliers, and standardizes data types.
*   **`data_cleaning_supplementary_data.py`**: Cleans the supplementary datasets from EPA (Fuel Economy) and NHTSA (Safety Ratings). Standardizes column names, selects relevant features, and handles missing values.
*   **`data_cleaning_user_reviews.py`**: Cleans the user review datasets sourced from Kaggle. Extracts relevant information like author, vehicle identifier, and rating, and standardizes the format.
*   **`feature_engineering.py`**: Performs feature engineering on the cleaned primary dataset. Creates new features like `vehicle_age` and `cylinders_numeric`. Integrates (merges) the cleaned supplementary datasets with the primary listings data based on common keys (year, make, model). Prepares the final feature set for modeling.
*   **`eda_analysis.py`**: Conducts Exploratory Data Analysis on the cleaned and potentially engineered dataset. Generates visualizations (histograms, scatter plots, box plots) and calculates descriptive statistics to understand data distributions and relationships (e.g., price vs. year/odometer, price by manufacturer).
*   **`machine_learning_modeling.py`**: Implements the machine learning pipeline.
    *   Splits data into training and testing sets.
    *   Applies preprocessing (imputation, scaling for numerical features, one-hot encoding for categorical features) using `ColumnTransformer`.
    *   Trains and tunes multiple regression models: Linear Regression, Random Forest Regressor, XGBoost Regressor, and a Feed-Forward Neural Network (TensorFlow/Keras).
    *   Evaluates models using R-squared and RMSE metrics.
    *   Generates diagnostic plots (Actual vs. Predicted, Residuals), Feature Importance plots (for tree-based models), and Partial Dependence Plots (PDPs).
*   **`requirements.txt`**: Lists the necessary Python libraries and their versions required to run the scripts.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Download Data:** Place the raw data files (from Kaggle, EPA, NHTSA) in an accessible location (e.g., a `data/raw/` directory). Paths within the scripts may need adjustment based on your data location.

## Usage

Run the scripts in the following sequence for the intended workflow:

1.  Execute the data cleaning scripts (`data_cleaning_...py`).
2.  Run the feature engineering script (`feature_engineering.py`) to combine datasets and create final features.
3.  Execute the EDA script (`eda_analysis.py`) to explore the prepared data.
4.  Run the modeling script (`machine_learning_modeling.py`) to train, evaluate, and analyze the predictive models.

Refer to individual scripts for specific input/output configurations.

## Key Results Summary

*   The tuned **XGBoost Regressor** achieved the best performance on the test set (R² ≈ 0.83, RMSE ≈ 0.366 on log-transformed price), explaining nearly 83% of the price variance.
*   Random Forest also performed well (Test R² ≈ 0.82).
*   Both ensemble models significantly outperformed the baseline Linear Regression (Test R² ≈ 0.63) and the specific Neural Network configuration (Test R² ≈ 0.72).
*   Feature importance analysis confirmed that `vehicle_age`, `odometer`, and `year` are dominant predictors, along with specific categorical features derived from `drive`, `fuel type`, `transmission`, `vehicle type`, and `condition`.

## Limitations

*   **Limited Temporal Scope:** The primary dataset covers only April-May 2021, preventing robust time-series analysis of market trends.
*   **Missing Data:** Significant imputation was required for several categorical features (e.g., `condition`, `cylinders`, `drive`), which might affect model accuracy.
*   **Potential Heteroscedasticity:** Residual plots suggest models might be less precise for higher-priced vehicles.

## Future Work

*   Acquire data with a broader and verified time range for temporal analysis.
*   Explore more sophisticated imputation techniques.
*   Further tune the Neural Network architecture and hyperparameters.
*   Utilize SHAP (SHapley Additive exPlanations) for deeper model interpretation.

---

*This project was completed as part of the ISYE 6740 course requirements.*
