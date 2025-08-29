import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
import joblib # For saving the model

# Import our custom modules
from data_processing import load_and_clean_data
from feature_engineering import create_features

def main():
    """Main function to run the model training pipeline."""
    # 1. Load and clean data
    raw_data_path = 'data/raw/patient_vitals.csv'
    clean_df = load_and_clean_data(raw_data_path)
    
    if clean_df is None:
        return

    # 2. Engineer features
    features_df = create_features(clean_df)
    
    # --- DUMMY TARGET VARIABLE CREATION (Replace with your real logic) ---
    # For this example, let's create a dummy 'deteriorated' label.
    # e.g., patients with high BP volatility are at risk.
    avg_std_sbp = features_df['std_sbp'].mean()
    features_df['deteriorated'] = (features_df['std_sbp'] > avg_std_sbp).astype(int)
    # --- END DUMMY TARGET ---

    # 3. Prepare data for modeling
    X = features_df.drop(columns=['Patient ID', 'deteriorated'])
    y = features_df['deteriorated']
    
    # 4. Split data
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=42, stratify=y
    )

    # 5. Train model
    print("Training XGBoost model...")
    model = XGBClassifier(eval_metric='logloss', random_state=42)
    model.fit(X_train, y_train)
    
    # 6. Evaluate model
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    auc_score = roc_auc_score(y_test, y_pred_proba)
    print(f"Model training complete. Test Set AUROC: {auc_score:.4f}")
    
    # 7. Save the model
    model_path = 'models/risk_model.joblib'
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    # To run the training, you will execute `python3 src/train_model.py` from the root directory
    main()