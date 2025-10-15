import streamlit as st
import pandas as pd
import pickle
import joblib
import os

# ----------------------------------------------------
# Load the trained model and scaler
# ----------------------------------------------------
MODEL_CANDIDATES = ["logistic_regression_model.pkl", "attrition_model.pkl", "employee-attrition.pkl", "attrition_model.joblib"]
SCALER_CANDIDATES = ["scaler.pkl", "scaler.joblib"]

model = None
scaler = None

# --- Load Model ---
for fname in MODEL_CANDIDATES:
    if os.path.exists(fname):
        if fname.endswith(".joblib"):
            model = joblib.load(fname)
        else:
            with open(fname, "rb") as f:
                model = pickle.load(f)
        break

if model is None:
    st.error("❌ No model file found. Please place one of these files in the folder:\n- logistic_regression_model.pkl\n- attrition_model.pkl\n- employee-attrition.pkl\n- attrition_model.joblib")
    st.stop()

# --- Load Scaler ---
for fname in SCALER_CANDIDATES:
    if os.path.exists(fname):
        if fname.endswith(".joblib"):
            scaler = joblib.load(fname)
        else:
            with open(fname, "rb") as f:
                scaler = pickle.load(f)
        break

if scaler is None:
    st.warning("⚠️ No scaler file found. Model input will not be scaled.")

# ----------------------------------------------------
# Streamlit page configuration
# ----------------------------------------------------
st.set_page_config(page_title="Employee Attrition Prediction", layout="wide")
st.title("💼 Employee Attrition Prediction App")

st.markdown("""
This app predicts whether an employee is likely to **leave (Attrition = Yes)** or **stay (Attrition = No)**  
based on their work-related information.
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
        # One-hot encode categorical variables if model expects them
        input_processed = pd.get_dummies(input_data)

        # Align columns with the model training data
        if hasattr(model, "feature_names_in_"):
            missing_cols = set(model.feature_names_in_) - set(input_processed.columns)
            for c in missing_cols:
                input_processed[c] = 0
            input_processed = input_processed[model.feature_names_in_]

        # Scale numerical features if scaler is present
        if scaler is not None:
            input_scaled = scaler.transform(input_processed)
        else:
            input_scaled = input_processed

        # Predict
        prediction = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0][1]

        if prediction == 1 or str(prediction).lower() == "yes":
            st.error(f"⚠️ The employee is **likely to leave**. (Probability: {prob:.2f})")
        else:
            st.success(f"✅ The employee is **likely to stay**. (Leave Probability: {prob:.2f})")

        st.write("---")
        st.subheader("🧠 Model Input Data")
        st.dataframe(input_data)

    except Exception as e:
        st.error(f"Prediction failed: {e}")
