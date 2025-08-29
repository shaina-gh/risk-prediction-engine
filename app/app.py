import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Function to generate dummy data
def generate_dummy_data():
    # Cohort data
    patient_ids = [f'PAT-{100+i}' for i in range(20)]
    data = {
        'Patient ID': patient_ids,
        'Risk Score': np.random.randint(5, 96, size=20),
        'Risk Trend': np.random.choice(['↑', '↓', '→'], size=20, p=[0.4, 0.4, 0.2]),
        'Top Risk Driver': np.random.choice([
            'BP Volatility', 'Medication Adherence', 'High Glucose Readings', 
            'Weight Gain', 'Low Activity Level'
        ], size=20)
    }
    cohort_df = pd.DataFrame(data)

    # Individual patient detail data (time series and explanations)
    patient_details = {}
    for pid in patient_ids:
        # Time series data for charts
        dates = pd.to_datetime(pd.date_range(end='2025-08-29', periods=90, freq='D'))
        bp_systolic = np.random.randint(110, 180, size=90)
        patient_details[pid] = {
            'time_series': pd.DataFrame({'Date': dates, 'Systolic BP': bp_systolic}),
            'explanations': {
                'increasing_risk': ['High BP Volatility (Std. Dev > 15mmHg)', 'Medication adherence at 65%'],
                'decreasing_risk': ['Consistent daily activity (avg. 6k steps)', 'Stable weight over last 30 days']
            },
            'recommendations': 'Schedule telehealth check-in to discuss blood pressure management.'
        }
    
    return cohort_df, patient_details

# --- Page Configuration ---
st.set_page_config(
    page_title="Chronic Care Risk Engine",
    page_icon="🩺",
    layout="wide"
)

# --- Data Loading ---
cohort_df, patient_details = generate_dummy_data()

# --- UI: Title & Filters ---
st.title("🩺 AI-Driven Risk Prediction Engine")
st.markdown("Dashboard for monitoring chronic care patients at risk of deterioration.")

st.sidebar.header("Filters")
risk_threshold = st.sidebar.slider(
    'Show patients with risk score above:',
    min_value=0, max_value=100, value=50, step=5
)

# --- UI: Cohort View Table ---
st.header("Patient Cohort View")

# Filter the dataframe based on the slider
filtered_df = cohort_df[cohort_df['Risk Score'] >= risk_threshold]

# Function to color the risk scores
def style_risk_score(score):
    if score > 70:
        color = 'red'
    elif score > 40:
        color = 'orange'
    else:
        color = 'green'
    return f'background-color: {color}; color: white'

# Display the styled dataframe
st.dataframe(
    filtered_df.style.applymap(style_risk_score, subset=['Risk Score']),
    use_container_width=True
)

# --- UI: Patient Detail View ---
st.header("Patient Detail View")

# Dropdown to select a patient from the filtered list
selected_patient_id = st.selectbox(
    'Select a Patient to View Details:',
    options=filtered_df['Patient ID'].unique()
)

if selected_patient_id:
    # Get the details for the selected patient
    details = patient_details[selected_patient_id]
    patient_cohort_info = cohort_df[cohort_df['Patient ID'] == selected_patient_id].iloc[0]

    # Display key metrics
    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            label="Current Risk Score",
            value=f"{patient_cohort_info['Risk Score']}%",
            delta=f"Trend: {patient_cohort_info['Risk Trend']}"
        )
    
    # Display the explainability and recommendations
    st.subheader("Key Risk Drivers & Recommendations")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🔴 Factors Increasing Risk")
        for factor in details['explanations']['increasing_risk']:
            st.markdown(f"- {factor}")
        
        st.markdown("#### 🟢 Factors Decreasing Risk")
        for factor in details['explanations']['decreasing_risk']:
            st.markdown(f"- {factor}")

    with col2:
        st.warning(f"**Recommended Next Action:**\n{details['recommendations']}")

    # Display the trend chart
    st.subheader("Blood Pressure Trend (Last 90 Days)")
    time_series_df = details['time_series']
    fig = px.line(
        time_series_df, 
        x='Date', 
        y='Systolic BP', 
        title='Systolic Blood Pressure Trend',
        labels={'Systolic BP': 'Systolic BP (mmHg)'}
    )
    st.plotly_chart(fig, use_container_width=True)