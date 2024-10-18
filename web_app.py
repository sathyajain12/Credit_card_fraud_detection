import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

# Load the trained model and scaler
model = joblib.load('fraud_detection_model.pkl')
scaler = joblib.load('scaler.pkl')

# Title of the web app
st.title('Real-time Credit Card Fraud Detection')

st.title("Credit Card Fraud Detection Model")
st.write("Enter the following features to check if the transaction is legitimate or fraudulent:")

# create input fields for user to enter feature values
input_df = st.text_input('Input All features')
input_df_lst = input_df.split(',')
# create a button to submit input and get prediction
submit = st.button("Submit")

if submit:
    try:
        # Get input feature values
        features = np.array(input_df_lst, dtype=np.float64)

        # Check if the number of features is correct
        expected_features = scaler.mean_.shape[0]  # Number of features expected by the scaler
        if features.shape[0] != expected_features:
            st.error(f"Expected {expected_features} features, but got {features.shape[0]}. Please check your input.")
        else:
            # Preprocess input data using the scaler
            features = scaler.transform(features.reshape(1, -1))
    
            # Make prediction
            prediction = model.predict(features)
            
            # Display result as a pop-up message
            if prediction[0] == 0:
                st.success("Legitimate transaction")
            else:
                st.error("Fraudulent transaction")
    except ValueError as e:
        st.error(f"Error in input data: {e}")