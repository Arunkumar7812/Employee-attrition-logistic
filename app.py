import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import os

# --- 1. CONFIGURATION ---
REQUIRED_COLS = [
    'Age', 'MonthlyIncome', 'TotalWorkingYears', 'YearsAtCompany',
    'EnvironmentSatisfaction',
    'OverTime_No', 'OverTime_Yes',
    'MaritalStatus_Divorced', 'MaritalStatus_Married', 'MaritalStatus_Single',
    'JobRole_Sales Representative', 'JobRole_Research Scientist',
    'JobRole_Laboratory Technician', 'JobRole_Sales Executive'
]

MODEL_PATH = 'logistic_regression_model.pkl'
SCALER_PATH = 'scaler.pkl'

# --- 2. MOCK SETUP (Run Once if files are missing) ---
def create_mock_files():
    st.info("Creating mock model and scaler files for demonstration. Replace them with your actual files for real use.")
    
    data = {}
    for col in ['Age', 'MonthlyIncome', 'TotalWorkingYears', 'YearsAtCompany', 'EnvironmentSatisfaction']:
        data[col] = np.random.rand(100) * 100
    for col in REQUIRED_COLS[5:]:
        data[col] = np.random.randint(0, 2, 100)
        
    X_mock = pd.DataFrame(data)
    y_mock = np.random.randint(0, 2, 100)
    
    mock_scaler = StandardScaler()
    X_scaled_mock = mock_scaler.fit_transform(X_mock)
    
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(mock_scaler, f)
        
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

# --- 4. STREAMLIT UI ---
st.set_page_config(page_title="Employee Attrition Risk Predictor", layout="centered")

st.markdown("""
<style>
.reportview-container .main { background-color: #f0f2f6; }
.stButton>button { background-color: #4CAF50; color: white; font-weight: bold; border-radius: 12px; padding: 10px 24px; border: none; transition: 0.3s; }
.stButton>button:hover { background-color: #45a049; }
</style>
""", unsafe_allow_html=True)

st.title("👨‍💼 Employee Attrition Risk Predictor")
st.markdown("Estimate the probability of an employee leaving the company based on key factors.")

# --- 5. SIDEBAR INPUTS ---
st.sidebar.header("Employee Profile Input")

age = st.sidebar.slider("Age", 18, 60, 30)
monthly_income = st.sidebar.number_input("Monthly Income", 1000, 50000, 5000)
total_working_years = st.sidebar.slider("Total Working Years", 0, 40, 5)
years_at_company = st.sidebar.slider("Years at Company", 0, 40, 3)
env_satisfaction = st.sidebar.slider("Environment Satisfaction (1-4)", 1, 4, 3)

overtime = st.sidebar.radio("OverTime", ("No", "Yes"))
overtime_no = int(overtime == "No")
overtime_yes = int(overtime == "Yes")

marital_status = st.sidebar.selectbox("Marital Status", ["Divorced", "Married", "Single"])
marital_divorced = int(marital_status == "Divorced")
marital_married = int(marital_status == "Married")
marital_single = int(marital_status == "Single")

job_role = st.sidebar.selectbox("Job Role", ["Sales Representative", "Research Scientist", "Laboratory Technician", "Sales Executive"])
job_sales_rep = int(job_role == "Sales Representative")
job_research = int(job_role == "Research Scientist")
job_lab_tech = int(job_role == "Laboratory Technician")
job_sales_exec = in
