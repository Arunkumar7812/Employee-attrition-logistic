import streamlit as st
import pickle
import pandas as pd
import numpy as np
import os

# --- Configuration (Loading both files) ---
MODEL_PATH = 'logistic_regression_model.pkl'
SCALER_PATH = 'logistic_regression_scaler.pkl' # Path for the preprocessor/scaler

# --- Utility Functions ---

@st.cache_resource
def load_artifacts():
    """Loads the pickled model and preprocessor (scaler) artifacts."""
    
    # Check if files exist
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        st.error(f"Required file(s) missing: '{MODEL_PATH}' and/or '{SCALER_PATH}'.")
        st.error("Please run the **'setup_model.py'** script first to create these files.")
        st.stop()

    try:
        # Load the Model (Logistic Regression object)
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)

        # Load the Preprocessor (ColumnTransformer object containing the StandardScaler)
        with open(SCALER_PATH, 'rb') as f:
            preprocessor = pickle.load(f)
            
        return model, preprocessor
        
    except Exception as e:
        st.error(f"Error loading model artifacts: {e}")
        st.stop()

# Load the model and preprocessor
model, preprocessor = load_artifacts()

# --- Streamlit App Layout ---

st.set_page_config(
    page_title="HR Attrition Risk Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("👨‍💼 Employee Attrition Risk Assessment")
st.markdown("Use this interactive tool to estimate an employee's risk of attrition based on key demographic and employment factors.")
st.divider()

# --- Sidebar for Input Features ---
st.sidebar.header("Employee Profile Input")

# Numerical features (using sliders for easy input)
age = st.sidebar.slider("Age", min_value=18, max_value=60, value=35)
monthly_income = st.sidebar.slider("Monthly Income ($)", min_value=1000, max_value=20000, value=7500, step=500)
years_at_company = st.sidebar.slider("Years at Company", min_value=0, max_value=40, value=5)

# Education level (assuming 1=Below College, 2=College, 3=Bachelor, 4=Master, 5=Doctorate)
education_map = {
    1: "Below College", 2: "College", 3: "Bachelor's Degree", 4: "Master's Degree", 5: "Doctorate"
}
education_label = st.sidebar.selectbox("Education Level", options=list(education_map.values()), index=2)
education = {v: k for k, v in education_map.items()}[education_label] # Map label back to integer

# Categorical features
job_role_options = ['Sales Executive', 'Research Scientist', 'Laboratory Technician', 'Manager', 'Manufacturing Director', 'Healthcare Representative', 'Sales Representative', 'Research Director', 'Human Resources']
job_role = st.sidebar.selectbox("Job Role", options=job_role_options, index=1)
gender = st.sidebar.radio("Gender", options=['Male', 'Female'], index=0, horizontal=True)

# --- Main Prediction Logic ---

# Create the input DataFrame
input_data = pd.DataFrame({
    'Age': [age],
    'MonthlyIncome': [monthly_income],
    'YearsAtCompany': [years_at_company],
    'JobRole': [job_role],
    'Gender': [gender],
    'Education': [education]
})

# Display the input data in the main area
st.subheader("Current Employee Profile")
st.dataframe(input_data, use_container_width=True)

# Perform Prediction
if st.button("Calculate Attrition Risk", type="primary"):
    with st.spinner("Analyzing profile..."):
        try:
            # 1. Transform the raw input data using the loaded preprocessor/scaler
            # This step performs scaling, encoding, etc.
            transformed_data = preprocessor.transform(input_data)

            # 2. Predict probability of attrition (class 1) using the model
            proba = model.predict_proba(transformed_data)[:, 1][0]
            risk_score = proba * 100
            
            # Define risk category based on a threshold
            if risk_score > 30:
                risk_level = "High Risk 🔴"
                st.balloons() 
            elif risk_score > 15:
                risk_level = "Medium Risk 🟡"
            else:
                risk_level = "Low Risk 🟢"

            # --- Results Display ---
            st.subheader("Prediction Result")
            col1, col2 = st.columns(2)

            with col1:
                st.metric(label="Calculated Attrition Risk Score", value=f"{risk_score:.2f}%")

            with col2:
                st.markdown(f"**Risk Level:**")
                st.markdown(f"## {risk_level}")
                
            st.progress(risk_score / 100, text="Risk Visualization")

            st.info(
                f"The model predicts a **{risk_score:.2f}%** probability of this employee leaving."
            )

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            st.warning("Ensure the input features match the model's training data structure.")

st.sidebar.markdown("---")
st.sidebar.caption("Model Artifacts: `logistic_regression_model.pkl` (Model) and `logistic_regression_scaler.pkl` (Preprocessor)")