import streamlit as st
import pandas as pd
import pickle
import joblib
import os
import re
import importlib

# ----------------------------------------------------
# Load the trained model and scaler
# ----------------------------------------------------

# Define candidates for model files (prioritize joblib as it's often better for large NumPy arrays)
MODEL_CANDIDATES = [
    "attrition_model.joblib",
    "employee-attrition.joblib",
    "logistic_regression_model.pkl", # Kept from original HEAD
    "attrition_model.pkl",
    "employee-attrition.pkl",
]
SCALER_CANDIDATES = ["scaler.joblib", "scaler.pkl"] # Prioritize joblib

model = None
scaler = None

# --- Load Model ---
model_found = None
for fname in MODEL_CANDIDATES:
    if os.path.exists(fname):
        model_found = fname
        break

if model_found is None:
    st.error("❌ No trained model found. Place one of the model files in the app folder: 'attrition_model.joblib', 'employee-attrition.joblib', 'logistic_regression_model.pkl', 'attrition_model.pkl', or 'employee-attrition.pkl'.")
    st.stop()

try:
    if model_found.endswith(".joblib"):
        # Attempt to load joblib file with a basic compatibility shim
        try:
            model = joblib.load(model_found)
        except AttributeError as e:
            # Simple fix for common sklearn versioning issues (as seen in the second version's logic)
            msg = str(e)
            m = re.search(r"Can't get attribute '(?P<attr>[^']+)' on <module '(?P<mod>[^']+)'", msg)
            if m:
                missing_attr = m.group('attr')
                missing_mod = m.group('mod')
                try:
                    # Create a simple placeholder class
                    mod = importlib.import_module(missing_mod)
                    placeholder = type(missing_attr, (), {})
                    setattr(mod, missing_attr, placeholder)
                    st.warning(f"Compatibility shim: created placeholder {missing_attr} in module {missing_mod}; retrying model load.")
                    model = joblib.load(model_found)
                except Exception as e2:
                    st.error(f"Failed to load joblib model due to AttributeError and failed shim: {e2}")
                    st.stop()
            else:
                st.error(f"Failed to load joblib model '{model_found}': {e}")
                st.stop()
    else:
        # Load .pkl files using pickle
        with open(model_found, "rb") as f:
            model = pickle.load(f)
except Exception as e:
    st.error(f"Failed to load model '{model_found}': {e}")
    st.stop()

# --- Load Scaler ---
scaler_found = None
for fname in SCALER_CANDIDATES:
    if os.path.exists(fname):
        scaler_found = fname
        break

if scaler_found is not None:
    try:
        if scaler_found.endswith(".joblib"):
            scaler = joblib.load(scaler_found)
        else:
            with open(scaler_found, "rb") as f:
                scaler = pickle.load(f)
    except Exception as e:
        st.error(f"Failed to load scaler '{scaler_found}': {e}")
        scaler = None # Set to None if loading fails, to proceed without scaling

if scaler is None:
    st.warning("⚠️ No scaler file found or failed to load. Model input will not be scaled. Proceeding with unscaled data.")


# ----------------------------------------------------
# Streamlit page configuration
# ----------------------------------------------------
st.set_page_config(page_title="Employee Attrition Prediction", layout="wide")
st.title("💼 Employee Attrition Prediction App")

st.markdown("""
This app predicts whether an employee is likely to **leave (Attrition = Yes)** or **stay (Attrition = No)** based on their work-related information.
""")

# ----------------------------------------------------
# Input Section
# ----------------------------------------------------
st.header("🧾 Enter Employee Details")

col1, col2, col3 = st.columns(3)

with col1:
    Age = st.number_input("Age", min_value=18, max_value=60, value=30)
    DailyRate = st.number_input("DailyRate", min_value=0, value=800)
    DistanceFromHome = st.number_input("DistanceFromHome", min_value=0, value=5)
    Education = st.selectbox("Education (1–5)", [1, 2, 3, 4, 5])
    EnvironmentSatisfaction = st.selectbox("Environment Satisfaction (1–4)", [1, 2, 3, 4])
    HourlyRate = st.number_input("HourlyRate", min_value=0, value=60)
    JobInvolvement = st.selectbox("Job Involvement (1–4)", [1, 2, 3, 4])
    JobLevel = st.selectbox("Job Level (1–5)", [1, 2, 3, 4, 5])

with col2:
    JobSatisfaction = st.selectbox("Job Satisfaction (1–4)", [1, 2, 3, 4])
    MonthlyIncome = st.number_input("Monthly Income", min_value=1000, max_value=20000, value=5000)
    MonthlyRate = st.number_input("Monthly Rate", min_value=1000, max_value=30000, value=10000)
    NumCompaniesWorked = st.number_input("Num Companies Worked", min_value=0, value=2)
    PercentSalaryHike = st.number_input("Percent Salary Hike", min_value=0, value=10)
    PerformanceRating = st.selectbox("Performance Rating (1–4)", [1, 2, 3, 4])
    RelationshipSatisfaction = st.selectbox("Relationship Satisfaction (1–4)", [1, 2, 3, 4])
    StockOptionLevel = st.selectbox("Stock Option Level (0–3)", [0, 1, 2, 3])

with col3:
    TotalWorkingYears = st.number_input("Total Working Years", min_value=0, value=5)
    TrainingTimesLastYear = st.number_input("Training Times Last Year", min_value=0, value=2)
    WorkLifeBalance = st.selectbox("Work Life Balance (1–4)", [1, 2, 3, 4])
    YearsAtCompany = st.number_input("Years At Company", min_value=0, value=3)
    YearsInCurrentRole = st.number_input("Years In Current Role", min_value=0, value=2)
    YearsSinceLastPromotion = st.number_input("Years Since Last Promotion", min_value=0, value=1)
    YearsWithCurrManager = st.number_input("Years With Current Manager", min_value=0, value=2)
    BusinessTravel = st.selectbox("Business Travel", ["Travel_Rarely", "Travel_Frequently", "Non-Travel"])
    Department = st.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
    EducationField = st.selectbox("Education Field", ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Human Resources", "Other"])
    Gender = st.selectbox("Gender", ["Male", "Female"])
    JobRole = st.selectbox("Job Role", [
        "Sales Executive", "Research Scientist", "Laboratory Technician",
        "Manufacturing Director", "Healthcare Representative",
        "Manager", "Sales Representative", "Research Director", "Human Resources"
    ])
    MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    OverTime = st.selectbox("OverTime", ["Yes", "No"])

# ----------------------------------------------------
# Prepare Input DataFrame
# ----------------------------------------------------
input_data = pd.DataFrame([{
    'Age': Age,
    'DailyRate': DailyRate,
    'DistanceFromHome': DistanceFromHome,
    'Education': Education,
    'EnvironmentSatisfaction': EnvironmentSatisfaction,
    'HourlyRate': HourlyRate,
    'JobInvolvement': JobInvolvement,
    'JobLevel': JobLevel,
    'JobSatisfaction': JobSatisfaction,
    'MonthlyIncome': MonthlyIncome,
    'MonthlyRate': MonthlyRate,
    'NumCompaniesWorked': NumCompaniesWorked,
    'PercentSalaryHike': PercentSalaryHike,
    'PerformanceRating': PerformanceRating,
    'RelationshipSatisfaction': RelationshipSatisfaction,
    'StockOptionLevel': StockOptionLevel,
    'TotalWorkingYears': TotalWorkingYears,
    'TrainingTimesLastYear': TrainingTimesLastYear,
    'WorkLifeBalance': WorkLifeBalance,
    'YearsAtCompany': YearsAtCompany,
    'YearsInCurrentRole': YearsInCurrentRole,
    'YearsSinceLastPromotion': YearsSinceLastPromotion,
    'YearsWithCurrManager': YearsWithCurrManager,
    'BusinessTravel': BusinessTravel,
    'Department': Department,
    'EducationField': EducationField,
    'Gender': Gender,
    'JobRole': JobRole,
    'MaritalStatus': MaritalStatus,
    'OverTime': OverTime
}])

# ----------------------------------------------------
# Make Prediction
# ----------------------------------------------------
if st.button("🔍 Predict Attrition"):
    try:
        # **Robust Data Processing (adopted from HEAD)**
        # 1. One-hot encode categorical variables
        input_processed = pd.get_dummies(input_data)

        # 2. Align columns with the model training data
        # This is CRITICAL. It ensures the input features are in the same order
        # as the features the model was trained on, adding missing OHE columns as 0.
        if hasattr(model, "feature_names_in_"):
            expected_cols = model.feature_names_in_
        # Fallback for models without feature_names_in_ (you might need to provide a list here)
        elif hasattr(model, "coef_") and isinstance(getattr(model, 'coef_'), pd.DataFrame):
            # Attempt to use coef_ if it's a DataFrame with feature names (less common)
            expected_cols = model.coef_.columns
        else:
             # This is a safe (but slow) assumption for many models; requires original training columns
            # For a real-world app, you should save the feature names along with the model/scaler.
            st.warning("Model does not expose feature names. Prediction might fail if features are misaligned.")
            # For demonstration, we'll try to proceed with the encoded data as-is if no feature names are found.
            expected_cols = input_processed.columns

        # Ensure all expected columns are present (fill with 0 if missing, drop extra)
        missing_cols = set(expected_cols) - set(input_processed.columns)
        for c in missing_cols:
            input_processed[c] = 0
        input_processed = input_processed[expected_cols]

        # 3. Scale numerical features if scaler is present
        if scaler is not None:
            # Note: The scaler should only be applied to the numerical columns.
            # If the entire one-hot encoded array is scaled, the OHE columns (0s and 1s) will be scaled too.
            # If the model was trained by scaling the full OHE array, this is correct.
            # Assuming the model expects the fully encoded array to be scaled, as per the original HEAD logic.
            input_scaled = scaler.transform(input_processed)
        else:
            input_scaled = input_processed

        # Predict
        prediction_output = model.predict(input_scaled)[0]
        # prob is the probability of the positive class (usually 1, or 'Yes' for attrition)
        prob = model.predict_proba(input_scaled)[0][1]

        # Check prediction type and report result
        # Standard binary prediction is 1/0 or 'Yes'/'No'
        if prediction_output in [1, "Yes"]:
            st.error(f"⚠️ The employee is **likely to leave**. (Probability: {prob:.2f})")
        else:
            st.success(f"✅ The employee is **likely to stay**. (Leave Probability: {prob:.2f})")

        st.write("---")
        st.subheader("🧠 Model Input Data")
        st.dataframe(input_data)

    except Exception as e:
        st.error(f"Prediction failed: {e}")
