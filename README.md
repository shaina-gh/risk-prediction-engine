# AI-Driven Risk Prediction Engine for Chronic Care Patients
This project is a prototype of an AI-driven tool designed to empower clinicians by proactively identifying chronic care patients at risk of deterioration. By leveraging historical patient data, the model predicts the likelihood of a significant health event within the next 90 days, providing explainable insights to guide early intervention.

# 🚀 Live Demo
You can view and interact with the live application here:
https://risk-prediction-engine.streamlit.app/

# Prediction Model Approach
Our approach frames this as a binary classification task. The goal is to predict one of two outcomes for a patient:

- Deterioration: The patient will experience a significant health event (e.g., unplanned hospitalization) within the next 90 days.

- Stable: The patient will not experience a significant health event in the next 90 days.

## We selected an XGBoost (Extreme Gradient Boosting) model, a powerful algorithm that excels at finding complex patterns in structured data like patient health records.

## How the Model Works: From Data to Prediction
The model transforms a patient's time-series data into a feature-rich snapshot for prediction.

- Defining the Prediction Window: The model looks back 30–180 days for input data and looks forward 90 days to determine the outcome (deterioration or stable).

- Feature Engineering: Instead of using raw data, the model calculates meaningful patterns from the historical window. For key metrics like blood pressure, we engineer features that describe its behavior:

- Aggregates: What is the patient’s average, maximum, and minimum reading?

- Volatility: What is the standard deviation? This measures the stability of a patient's condition.

- Trends: What is the slope of the readings over time? This shows if the patient is consistently getting better or worse.

- Making a Prediction: This rich set of engineered features is fed into the trained XGBoost model, which outputs a single probability score from 0% to 100%.

# Model Evaluation Metrics
We use a suite of metrics to ensure the model is both accurate and trustworthy in a clinical setting.

- AUROC (Area Under the ROC Curve): Measures the model's ability to distinguish between at-risk and stable patients.

- AUPRC (Area Under the Precision-Recall Curve): Crucial for imbalanced datasets, as it evaluates the precision of high-risk alerts.

- Calibration Plot: Verifies that the model's predicted probabilities are accurate (e.g., a 70% risk score means a ~70% chance of an event).

- Confusion Matrix: Provides a detailed breakdown of the model's errors, especially False Negatives (at-risk patients the model missed).

## What Factors Drive the Model’s Predictions?
To ensure our model is not a "black box," we use SHAP (SHapley Additive exPlanations) to make its predictions fully transparent.

## Global Explanations (Overall Trends)
At a high level, we can see which factors are most important across all patients. The model might learn that blood pressure volatility is the most significant predictor overall, followed by poor medication adherence.

## Local Explanations (Patient-Level)
More importantly, we can explain every prediction for each individual patient. The dashboard translates these factors into simple, actionable terms. 

For example:
"Jane Doe's risk score is 82%. The primary factors increasing her risk are:
- High volatility in her morning glucose readings.
- A missed refill for her heart medication.
- A factor decreasing her risk is her consistent daily step count."

# 🛠 Setup and Local Usage
To run this project on your own machine, follow these steps.

- Clone the repository:
git clone [https://github.com/shaina-gh/risk-prediction-engine.git](https://github.com/shaina-gh/risk-prediction-engine.git)

cd risk-prediction-engine

## Install system dependencies (for Git LFS):
### On macOS
brew install git-lfs
### On Debian/Ubuntu
sudo apt-get install git-lfs

Then, initialize it in the repository:
git lfs install

Install Python dependencies:
pip install -r requirements.txt

Train the model:
python3 src/train_model.py

Run the Streamlit app:
streamlit run app.py
