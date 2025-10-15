import streamlit as st
import pandas as pd
import pickle
import numpy as np

# --- 1. Load the Model ---
MODEL_PATH = 'employee-attrition.pkl'
try:
    with open(MODEL_PATH, 'rb') as file:
        model = pickle.load(file)
    st.success(f"Model loaded successfully from {MODEL_PATH}")
except FileNotFoundError:
    st.error(f"Error: Model file '{MODEL_PATH}' not found. Please ensure it is in the same directory.")
    model = None
except Exception as e:
    st.error(f"Error loading model: {e}")
    model = None

# --- 2. Configuration and Helper Data ---

# Define the full set of 30 features required by the trained model pipeline.
# Grouped features for better UI flow in the sidebar.
FEATURE_DEFS = {
    # Personal & Demographic (7 features)
    'Age': {'type': 'slider', 'min': 18, 'max': 60, 'default': 35},
    'Gender': {'type': 'select', 'options': ['Male', 'Female'], 'default': 'Male'},
    'MaritalStatus': {'type': 'select', 'options': ['Single', 'Married', 'Divorced'], 'default': 'Married'},
    'DistanceFromHome': {'type': 'slider', 'min': 1, 'max': 29, 'default': 10},
    'NumCompaniesWorked': {'type': 'slider', 'min': 0, 'max': 9, 'default': 2},
    'Education': {'type': 'select', 'options': [1, 2, 3, 4, 5], 'default': 3, 'help': '1=Below College, 5=Doctorate'},
    'EducationField': {'type': 'select', 'options': ['Life Sciences', 'Medical', 'Marketing', 'Technical Degree', 'Human Resources', 'Other'], 'default': 'Life Sciences'},

    # Role & Travel (5 features)
    'Department': {'type': 'select', 'options': ['Research & Development', 'Sales', 'Human Resources'], 'default': 'Research & Development'},
    'JobRole': {'type': 'select', 'options': ['Sales Executive', 'Research Scientist', 'Laboratory Technician', 'Manufacturing Director', 'Healthcare Representative', 'Manager', 'Sales Representative', 'Research Director', 'Human Resources'], 'default': 'Research Scientist'},
    'JobLevel': {'type': 'select', 'options': [1, 2, 3, 4, 5], 'default': 2, 'help': '1=Entry Level, 5=Executive'},
    'OverTime': {'type': 'select', 'options': ['Yes', 'No'], 'default': 'No'},
    'BusinessTravel': {'type': 'select', 'options': ['Non-Travel', 'Travel_Rarely', 'Travel_Frequently'], 'default': 'Travel_Rarely'},

    # Compensation & Financial (6 features)
    'MonthlyIncome': {'type': 'slider', 'min': 1000, 'max': 20000, 'default': 5000, 'step': 100},
    'DailyRate': {'type': 'slider', 'min': 100, 'max': 1500, 'default': 800, 'step': 50},
    'HourlyRate': {'type': 'slider', 'min': 30, 'max': 100, 'default': 65},
    'MonthlyRate': {'type': 'slider', 'min': 2000, 'max': 27000, 'default': 14000, 'step': 100},
    'PercentSalaryHike': {'type': 'slider', 'min': 11, 'max': 25, 'default': 15},
    'StockOptionLevel': {'type': 'select', 'options': [0, 1, 2, 3], 'default': 1},

    # Experience & Tenure (5 features)
    'TotalWorkingYears': {'type': 'slider', 'min': 0, 'max': 40, 'default': 10},
    'YearsAtCompany': {'type': 'slider', 'min': 0, 'max': 40, 'default': 5},
    'YearsInCurrentRole': {'type': 'slider', 'min': 0, 'max': 18, 'default': 3},
    'YearsSinceLastPromotion': {'type': 'slider', 'min': 0, 'max': 15, 'default': 1},
    'YearsWithCurrManager': {'type': 'slider', 'min': 0, 'max': 17, 'default': 3},

    # Satisfaction & Ratings (7 features)
    'JobSatisfaction': {'type': 'select', 'options': [1, 2, 3, 4], 'default': 3, 'help': '1=Low, 4=High'},
    'EnvironmentSatisfaction': {'type': 'select', 'options': [1, 2, 3, 4], 'default': 3, 'help': '1=Low, 4=High'},
    'RelationshipSatisfaction': {'type': 'select', 'options': [1, 2, 3, 4], 'default': 3, 'help': '1=Low, 4=High'},
    'JobInvolvement': {'type': 'select', 'options': [1, 2, 3, 4], 'default': 3, 'help': '1=Low, 4=High'},
    'WorkLifeBalance': {'type': 'select', 'options': [1, 2, 3, 4], 'default': 3, 'help': '1=Bad, 4=Best'},
    'PerformanceRating': {'type': 'select', 'options': [1, 2, 3, 4], 'default': 3, 'help': '1=Low, 4=High'},
    'TrainingTimesLastYear': {'type': 'select', 'options': [0, 1, 2, 3, 4, 5, 6], 'default': 3},
}


# Placeholder for Feature Importance (Replace with actual data from your notebook's Cell 35)
# This mock data is just to show how the display works.
FEATURE_COEFFICIENTS = pd.DataFrame({
    'Feature': [
        'OverTime_Yes', 'MaritalStatus_Single', 'JobRole_Sales Representative', 
        'Department_Human Resources', 'DistanceFromHome', 'TotalWorkingYears', 
        'JobSatisfaction_4', 'MonthlyIncome', 'Age', 'YearsAtCompany'
    ],
    'Coefficient': [
        2.5, 1.8, 1.5, 1.2, 0.8, -1.5, -1.8, -2.0, -2.5, -2.8
    ]
}).sort_values(by='Coefficient', ascending=False).set_index('Feature')


# --- 3. Streamlit App Layout and Logic ---

def get_user_input():
    """Collects user input from the sidebar based on FEATURE_DEFS."""
    input_data = {}
    
    st.sidebar.markdown("### Employee Attributes")
    
    # Iterate through the DEFINED order to ensure the DataFrame columns match the model
    for feature, config in FEATURE_DEFS.items():
        key = feature.replace(' ', '_')
        
        if config['type'] == 'slider':
            input_data[feature] = st.sidebar.slider(
                feature, 
                min_value=config['min'], 
                max_value=config['max'], 
                value=config['default'], 
                step=config.get('step', 1)
            )
        elif config['type'] == 'select':
            input_data[feature] = st.sidebar.selectbox(
                feature, 
                options=config['options'], 
                index=config['options'].index(config['default']),
                help=config.get('help', None)
            )
            
    # Convert input to a single-row DataFrame. The column order is implicitly 
    # preserved from the FEATURE_DEFS dictionary.
    return pd.DataFrame([input_data])

def display_feature_importance(df):
    """Displays the top factors from the coefficient analysis."""
    st.markdown("---")
    st.header("Model Insights: Feature Importance")
    st.markdown("""
        The factors below show the effect of each feature on the likelihood of Attrition.
        (Note: These are sample values. **Replace these with the actual coefficients from your model's `coef_df`**.)
    """)

    top_increase = df.head(5).reset_index()
    top_decrease = df.tail(5).sort_values(by='Coefficient', ascending=True).reset_index()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Top Factors Increasing Attrition Likelihood (+)")
        st.dataframe(
            top_increase[['Feature', 'Coefficient']].style.background_gradient(cmap='Reds'),
            use_container_width=True,
            hide_index=True
        )

    with col2:
        st.subheader("Top Factors Decreasing Attrition Likelihood (-)")
        st.dataframe(
            top_decrease[['Feature', 'Coefficient']].style.background_gradient(cmap='Blues_r'),
            use_container_width=True,
            hide_index=True
        )


def main():
    """Main function to run the Streamlit application."""
    st.set_page_config(page_title="Employee Attrition Predictor", layout="wide")
    
    # Title and Introduction
    st.title("Employee Attrition Prediction App")
    st.markdown("### Predict the likelihood of an employee leaving the company.")
    st.markdown("Adjust the employee's attributes in the sidebar on the left and click 'Predict'.")

    # Get User Input
    input_df = get_user_input()

    # Prediction Logic
    if model:
        st.header("Prediction Result")
        
        # Prepare data for display before prediction
        st.markdown("#### Input Data Preview")
        st.dataframe(input_df)

        if st.button("Predict Attrition Likelihood"):
            try:
                # Prediction
                # This will now include all 30 features expected by the pipeline
                prediction_proba = model.predict_proba(input_df)
                
                # Attrition Yes Probability is usually the second column (index 1)
                attrition_proba = prediction_proba[0, 1] 
                
                # Format output
                percentage = f"{attrition_proba * 100:.2f}%"
                
                # Display result with a gauge or progress bar
                st.metric(
                    label="Probability of Attrition (Yes)", 
                    value=percentage, 
                    delta=f"{attrition_proba:.2f} (Base value)" if attrition_proba < 0.5 else f"{attrition_proba:.2f} (Warning)"
                )
                
                if attrition_proba >= 0.5:
                    st.warning("⚠️ High Risk of Attrition!")
                else:
                    st.success("✅ Low Risk of Attrition.")

                # Display Feature Importance based on the mock/actual coefficient data
                display_feature_importance(FEATURE_COEFFICIENTS)


            except Exception as e:
                # General error handling remains useful
                st.error(f"Prediction failed. This usually means the input features or their order/types do not match what the loaded model expects. Error: {e}")
                st.exception(e)
                
    else:
        st.warning("Cannot run prediction because the model failed to load. Please check the error message above.")


if __name__ == "__main__":
    main()
