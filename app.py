import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import requests
import os

# Import your custom processing functions from the src folder
# Vercel needs to know the path starts from the root
from src.data_processing import load_and_clean_data
from src.feature_engineering import create_features

# --- Page Configuration ---
st.set_page_config(
    page_title="Chronic Care Risk Engine",
    page_icon="🩺",
    layout="wide"
)

# --- Model & Data Loading ---
# This decorator caches the model, so it's only downloaded ONCE when the app starts.
# This is the new, more robust function
@st.cache_resource
def load_model_from_url():
    model_url = "https://media.githubusercontent.com/media/shaina-gh/risk-prediction-engine/main/models/risk_model.joblib"
    model_path = "/tmp/risk_model.joblib"
    
    # Download the file only if it doesn't already exist in the temporary folder
    if not os.path.exists(model_path):
        with st.spinner("Downloading model... this may take a moment."):
            try:
                response = requests.get(model_url, allow_redirects=True)
                # Raise an exception if the download fails (e.g., 404 error)
                response.raise_for_status()
                
                with open(model_path, 'wb') as f:
                    f.write(response.content)
            except requests.exceptions.RequestException as e:
                st.error(f"Error downloading model: {e}")
                # Stop the app if the model can't be downloaded
                st.stop()
    
    # Load the model from the temporary path
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

@st.cache_data
def process_data(_model): # Pass the model into the function
    # Load and process the patient data using your pipeline
    raw_df = load_and_clean_data('data/raw/patient_vitals.csv')
    features_df = create_features(raw_df)
    
    # Ensure features are in the same order as during training
    X = features_df.drop(columns=['Patient ID'])
    
    # Make predictions
    risk_scores = _model.predict_proba(X)[:, 1]
    
    # Create the cohort dataframe for the dashboard
    cohort_df = features_df[['Patient ID']].copy()
    cohort_df['Risk Score'] = (risk_scores * 100).astype(int)
    
    # Dummy data for trend and drivers
    cohort_df['Risk Trend'] = '→'
    cohort_df['Top Risk Driver'] = 'BP Volatility'
    
    return cohort_df, features_df

# Main app execution
model = load_model_from_url()
cohort_df, features_df = process_data(model)


# --- UI: Title & Filters ---
st.title("🩺 AI-Driven Risk Prediction Engine")
st.markdown("This dashboard is powered by a model trained on patient vital signs.")

st.sidebar.header("Filters")
risk_threshold = st.sidebar.slider(
    'Show patients with risk score above:',
    min_value=0, max_value=100, value=50, step=5
)

# --- UI: Cohort View Table ---
st.header("Patient Cohort View")
filtered_df = cohort_df[cohort_df['Risk Score'] >= risk_threshold]

def style_risk_score(score):
    if score > 70: color = 'red'
    elif score > 40: color = 'orange'
    else: color = 'green'
    return f'background-color: {color}; color: white'

st.dataframe(
    filtered_df.style.applymap(style_risk_score, subset=['Risk Score']),
    use_container_width=True
)

# --- UI: Patient Detail View ---
st.header("Patient Detail View")

if not filtered_df.empty:
    selected_patient_id = st.selectbox(
        'Select a Patient to View Details:',
        options=filtered_df['Patient ID'].unique()
    )

    if selected_patient_id:
        patient_cohort_info = filtered_df[filtered_df['Patient ID'] == selected_patient_id].iloc[0]
        patient_feature_info = features_df[features_df['Patient ID'] == selected_patient_id].iloc[0]

        st.metric(
            label="Current Risk Score",
            value=f"{patient_cohort_info['Risk Score']}%"
        )
        
        st.subheader(f"Feature Values for {selected_patient_id}")
        st.write(patient_feature_info) # Display all feature values for the selected patient
        
        # TODO: Replace the dummy text below with real SHAP explanations
        st.subheader("Key Risk Drivers")
        st.warning("**Note:** The explanations below are placeholders. A real implementation would use SHAP values calculated for this specific patient.")
        st.markdown("- **High `std_sbp` (Blood Pressure Volatility):** This is the primary factor increasing risk according to the model's logic.")
        st.markdown("- **High `max_hr` (Maximum Heart Rate):** Elevated peak heart rates are contributing to the risk score.")

else:
    st.warning("No patients match the current filter settings.")