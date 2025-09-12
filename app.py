import streamlit as st
import pandas as pd
import joblib

# Set the title and a brief description for your app
st.title("Telecom Customer Churn Prediction")
st.markdown("This app predicts whether a customer is likely to churn based on their information.")

# Define the filename of the saved pipeline
model_filename = 'churn_model_pipeline.pkl'

# Check if the model file exists and load it
try:
    if not joblib:
        st.error("Error: Joblib library is not installed.")
    elif not pd:
        st.error("Error: Pandas library is not installed.")
    elif not st:
        st.error("Error: Streamlit library is not installed.")
    elif not os.path.exists(model_filename):
        st.error(f"Error: The model file '{model_filename}' was not found. "
                 "Please make sure you have run the training script to save the model.")
    else:
        loaded_model = joblib.load(model_filename)
        st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"An error occurred while loading the model: {e}")
    st.stop()

# Create a Streamlit form for user input
with st.form(key='churn_prediction_form'):
    st.header("Customer Information")
    
    # Input fields for each feature
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", ('Male', 'Female'))
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        tenure = st.number_input("Tenure in Months", min_value=1, max_value=72, value=12)
        contract = st.selectbox("Contract Type", ('Month-to-Month', 'One-Year', 'Two-Year'))
        monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=50.0)
    
    with col2:
        internet_service = st.selectbox("Internet Service", ('FiberOptic', 'DSL', 'No'))
        tech_support = st.selectbox("Tech Support", ('Yes', 'No'))
        online_security = st.selectbox("Online Security", ('Yes', 'No'))
        payment_method = st.selectbox("Payment Method", ('Cash', 'BankTransfer', 'CreditCard', 'EWallet'))
        complaints = st.selectbox("Complaints", ('Yes', 'No'))
        total_charges = st.number_input("Total Charges", min_value=0.0, value=600.0)

    # Submit button
    submit_button = st.form_submit_button(label='Predict Churn')

# When the form is submitted
if submit_button:
    # Create a DataFrame from the user input
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

    user_input_df = pd.DataFrame(user_input_data)

    # Make a prediction using the loaded model pipeline
    try:
        prediction = loaded_model.predict(user_input_df)
        
        # Display the result
        st.subheader("Prediction Result")
        if prediction[0] == 'Yes':
            st.error("The model predicts this customer is LIKELY TO CHURN.")
        else:
            st.success("The model predicts this customer is UNLIKELY TO CHURN.")
            
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
