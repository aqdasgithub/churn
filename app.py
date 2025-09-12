import streamlit as st
import pandas as pd
import joblib
import os

# Define the filename of the saved pipeline
MODEL_FILENAME = 'churn_model_pipeline.pkl'

# Set up the Streamlit page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📈",
    layout="centered"
)

# --- Debug: Show current working directory & files ---
st.write("📂 Current working directory:", os.getcwd())
st.write("📂 Files in this directory:", os.listdir())

# --- Load the model ---
@st.cache_resource
def load_model():
    """Loads the pre-trained churn model pipeline."""
    if not os.path.exists(MODEL_FILENAME):
        return None
    try:
        loaded_model = joblib.load(MODEL_FILENAME)
        return loaded_model
    except Exception:
        return None

# Load the model at the start of the app
loaded_model = load_model()

# --- Main App ---
if loaded_model is None:
    st.error(f"❌ The file '{MODEL_FILENAME}' was not found or could not be loaded.")
    st.info("Please make sure you have trained and saved the model pipeline as 'churn_model_pipeline.pkl'.")
else:
    st.title("📊 Customer Churn Prediction")
    st.markdown("### Please enter the customer's details to predict if they will churn.")

    # --- Create input widgets for user data ---
    with st.form("customer_input_form"):
        # Categorical inputs
        col1, col2, col3 = st.columns(3)
        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
        with col2:
            contract = st.selectbox("Contract Type", ["Month-to-Month", "One-Year", "Two-Year"])
        with col3:
            internet_service = st.selectbox("Internet Service", ["FiberOptic", "DSL", "No"])
        
        col4, col5 = st.columns(2)
        with col4:
            tech_support = st.selectbox("Tech Support", ["Yes", "No"])
        with col5:
            online_security = st.selectbox("Online Security", ["Yes", "No"])

        payment_method = st.selectbox("Payment Method", ["Cash", "BankTransfer", "CreditCard", "EWallet"])
        complaints = st.selectbox("Complaints", ["Yes", "No"])

        # Numerical inputs
        st.markdown("---")
        st.subheader("Numerical Inputs")
        age = st.number_input("Age (in years)", min_value=18, max_value=100, value=35)
        tenure = st.number_input("Tenure (in months)", min_value=0, max_value=100, value=24)
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=50.0)
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=1200.0)

        # Submit button for the form
        submitted = st.form_submit_button("Predict Churn")

    if submitted:
        # Create a dictionary from the user input
        user_input_data = {
            'Gender': [gender],
            'Age': [age],
            'Tenure_Months': [tenure],
            'ContractType': [contract],
            'MonthlyCharges': [monthly_charges],
            'InternetService': [internet_service],
            'TechSupport': [tech_support],
            'OnlineSecurity': [online_security],
            'PaymentMethod': [payment_method],
            'Complaints': [complaints],
            'TotalCharges': [total_charges]
        }

        # Convert to DataFrame
        user_input_df = pd.DataFrame(user_input_data)

        # Make a prediction
        st.subheader("Prediction Result")
        try:
            prediction = loaded_model.predict(user_input_df)

            # Handle both string and numeric outputs
            pred_value = prediction[0]
            if pred_value in ['Yes', 1]:
                st.warning("🚨 The model predicts this customer is **LIKELY TO CHURN**.")
            else:
                st.success("✅ The model predicts this customer is **UNLIKELY TO CHURN**.")

            # Also show probability if available
            if hasattr(loaded_model, "predict_proba"):
                proba = loaded_model.predict_proba(user_input_df)[0][1]  # probability of churn
                st.info(f"📊 Churn Probability: **{proba:.2%}**")

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            st.info("Please ensure the model file is valid and the input data matches the expected format.")
