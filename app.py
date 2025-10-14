import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import os

# --- 1. CONFIGURATION ---

# CRITICAL FIX: The REQUIRED_COLS list is expanded to include all 9 common JobRole
# categories to prevent Feature Name Mismatch error during scaling.
REQUIRED_COLS = [
    'Age', 'MonthlyIncome', 'TotalWorkingYears', 'YearsAtCompany',
    'EnvironmentSatisfaction',
    'OverTime_No', 'OverTime_Yes',
    'MaritalStatus_Divorced', 'MaritalStatus_Married', 'MaritalStatus_Single',
    # All 9 common Job Roles (OHE features)
    'JobRole_Sales Representative', 'JobRole_Research Scientist',
    'JobRole_Laboratory Technician', 'JobRole_Sales Executive',
    'JobRole_Manufacturing Director', 'JobRole_Healthcare Representative',
    'JobRole_Human Resources', 'JobRole_Manager', 'JobRole_Research Director'
]

MODEL_PATH = 'logistic_regression_model.pkl'
SCALER_PATH = 'scaler.pkl'

# --- 2. MOCK SETUP (Run Once if files are missing) ---
# This ensures the app is runnable even without the user's saved files.
# Users should replace model.pkl and scaler.pkl with their actual files.
def create_mock_files():
    """Creates mock model and scaler files for initial run."""
    st.info("Creating mock model and scaler files for demonstration. Please replace them with your actual model.pkl and scaler.pkl for real use.")

    # 1. Generate dummy data for training (must match REQUIRED_COLS)
    data = {}
    for col in ['Age', 'MonthlyIncome', 'TotalWorkingYears', 'YearsAtCompany', 'EnvironmentSatisfaction']:
        data[col] = np.random.rand(100) * 100
    for col in REQUIRED_COLS[5:]:
        data[col] = np.random.randint(0, 2, 100)

    X_mock = pd.DataFrame(data, columns=REQUIRED_COLS) # Explicitly setting columns
    y_mock = np.random.randint(0, 2, 100) # Target: Attrition (0=No, 1=Yes)

    # 2. Train and save mock scaler
    mock_scaler = StandardScaler()
    X_scaled_mock = mock_scaler.fit_transform(X_mock)
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(mock_scaler, f)

    # 3. Train and save mock model
    mock_model = LogisticRegression(solver='liblinear')
    mock_model.fit(X_scaled_mock, y_mock)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(mock_model, f)
    st.success("Mock model and scaler created!")


if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    create_mock_files()

# --- 3. LOAD MODEL AND SCALER ---

try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
except Exception as e:
    st.error(f"Error loading model or scaler files: {e}")
    st.stop()


# --- 4. STREAMLIT UI AND LOGIC ---

st.set_page_config(page_title="Employee Attrition Risk Predictor", layout="centered")

st.markdown("""
    <style>
    .reportview-container .main {
        background-color: #f0f2f6;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 12px;
        padding: 10px 24px;
        border: none;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    </style>
""", unsafe_allow_html=True)

st.title("👨‍💼 Employee Attrition Risk Predictor")
st.markdown("Use this tool to estimate the probability of an employee leaving the company based on key factors identified by the Logistic Regression model.")

# --- Sidebar for Input ---
st.sidebar.header("Employee Profile Input")

# Numerical Inputs
age = st.sidebar.slider("Age (Years)", 18, 60, 30)
monthly_income = st.sidebar.number_input("Monthly Income (USD)", 1000, 20000, 5000)
total_years = st.sidebar.slider("Total Working Years", 0, 40, 5)
years_at_company = st.sidebar.slider("Years at Company", 0, 40, 3)

# Ordinal/Categorical Mapped to Numerical
environment_satisfaction_map = {
    "1 - Low": 1, "2 - Medium": 2, "3 - High": 3, "4 - Very High": 4
}
env_satisfaction_label = st.sidebar.selectbox(
    "Environment Satisfaction",
    options=list(environment_satisfaction_map.keys()),
    index=2
)
env_satisfaction = environment_satisfaction_map[env_satisfaction_label]

# Categorical Inputs
overtime = st.sidebar.selectbox("OverTime", options=['Yes', 'No'])
marital_status = st.sidebar.selectbox("Marital Status", options=['Single', 'Married', 'Divorced'])
job_role = st.sidebar.selectbox(
    "Job Role",
    options=[
        'Sales Representative', 'Research Scientist', 'Laboratory Technician',
        'Sales Executive', 'Manufacturing Director', 'Healthcare Representative',
        'Human Resources', 'Manager', 'Research Director' # <--- FIX: Added missing Job Roles
    ]
)

# --- Main Page Prediction Button ---
if st.button("Predict Attrition Risk"):
    # 1. Prepare Input Data Dictionary
    # FIX: Initializing ALL 9 JobRole OHE columns to 0.
    input_data = {
        'Age': age,
        'MonthlyIncome': monthly_income,
        'TotalWorkingYears': total_years,
        'YearsAtCompany': years_at_company,
        'EnvironmentSatisfaction': env_satisfaction,
        'OverTime_No': 0, 'OverTime_Yes': 0,
        'MaritalStatus_Divorced': 0, 'MaritalStatus_Married': 0, 'MaritalStatus_Single': 0,
        'JobRole_Sales Representative': 0, 'JobRole_Research Scientist': 0,
        'JobRole_Laboratory Technician': 0, 'JobRole_Sales Executive': 0,
        'JobRole_Manufacturing Director': 0, 'JobRole_Healthcare Representative': 0,
        'JobRole_Human Resources': 0, 'JobRole_Manager': 0, 'JobRole_Research Director': 0 # <--- FIX: Initialized the full set
    }

    # 2. Apply One-Hot Encoding to Categorical Inputs
    if f'OverTime_{overtime}' in input_data:
        input_data[f'OverTime_{overtime}'] = 1

    if f'MaritalStatus_{marital_status}' in input_data:
        input_data[f'MaritalStatus_{marital_status}'] = 1

    # Apply only for the role selected
    role_col = f'JobRole_{job_role}'
    if role_col in input_data:
        input_data[role_col] = 1


    # 3. Create DataFrame and enforce column order
    # The dataframe will now always have 19 features in the correct order.
    try:
        input_df = pd.DataFrame([input_data])
        # Reindex to ensure all columns exist and are in the correct order
        input_df = input_df.reindex(columns=REQUIRED_COLS, fill_value=0)
    except Exception as e:
        st.error(f"Error structuring input data: {e}")
        st.stop()


    # 4. Scale the input data
    input_scaled = scaler.transform(input_df)

    # 5. Make Prediction
    # Prediction Probability (P(Attrition=1))
    prediction_proba = model.predict_proba(input_scaled)[:, 1][0]
    risk_percent = round(prediction_proba * 100, 2)

    st.subheader("Prediction Result")

    # Display results with dynamic styling
    if risk_percent > 70:
        st.error(f"🔴 High Attrition Risk: {risk_percent}% Probability")
        st.markdown("**Recommendation:** Immediate intervention and retention strategy required.")
    elif risk_percent > 40:
        st.warning(f"🟡 Moderate Attrition Risk: {risk_percent}% Probability")
        st.markdown("**Recommendation:** Monitor closely and conduct proactive stay interviews.")
    else:
        st.success(f"🟢 Low Attrition Risk: {risk_percent}% Probability")
        st.markdown("**Recommendation:** This employee is likely stable. Continue standard engagement practices.")

    # Show the underlying data for transparency (optional)
    with st.expander("Show detailed prediction features"):
        st.dataframe(input_df.iloc[0])

# --- Footer ---
st.markdown(
    """
    ---
    *Disclaimer: This prediction is based on a Logistic Regression model and historical data, and should be used as an indicator, not a definitive forecast.*
    """
)
