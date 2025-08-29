import pandas as pd
import numpy as np

def create_features(df):
    """
    Engineers features from the cleaned time-series data for each patient.
    
    Args:
        df (pd.DataFrame): The cleaned data from load_and_clean_data.
        
    Returns:
        pd.DataFrame: A DataFrame with one row per patient and engineered features.
    """
    # Ensure data is sorted by patient and date for correct calculations
    df = df.sort_values(by=['Patient ID', 'date'])
    
    # Group by patient to calculate features
    grouped = df.groupby('Patient ID')
    
    features = grouped.agg(
        # Aggregate features
        avg_hr=('heart_rate', 'mean'),
        max_hr=('heart_rate', 'max'),
        avg_sbp=('systolic_bp', 'mean'),
        max_sbp=('systolic_bp', 'max'),
        
        # Volatility features
        std_hr=('heart_rate', 'std'),
        std_sbp=('systolic_bp', 'std')
    ).reset_index()

    # Trend feature (slope of blood pressure over time)
    def calculate_trend(series):
        # Fit a line (y = mx + c) and return the slope 'm'
        y = series.values
        x = np.arange(len(y))
        # Find the slope of the linear regression line
        slope, _ = np.polyfit(x, y, 1)
        return slope

    trends = grouped['systolic_bp'].apply(calculate_trend).reset_index(name='sbp_trend')
    
    # Merge trend features back into the main feature set
    features = pd.merge(features, trends, on='Patient ID')
    
    # Fill any potential NaNs created by std dev on single-entry groups
    features = features.fillna(0)
    
    print("Feature engineering complete.")
    return features