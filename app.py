import streamlit as st
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# Load the model, encoders and scaler
with open('pickle/random_forest.pkl', 'rb') as file:
    rf_model = pickle.load(file)

with open('pickle/label_encoder.pkl', 'rb') as file:
    label_encoder = pickle.load(file)

with open('pickle/scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# create streamlit app
st.title('Customer Churn Prediction')

# create user input
Gender = st.selectbox('Gender', ["Male","Female"])

Age = st.number_input('Age', value=0)

Balance = st.number_input('Balance', min_value=0, step=1000)

Credit_Score = st.number_input('Credit Score', 300, 900)

Salary = st.number_input('Salary', min_value=0, step=2000)

Tenure = st.number_input('Tenure', 0, 10)

Number_of_products = st.number_input('Number of Products', 1, 4)

Has_credit_card = st.selectbox('Credit Card', ['Yes', 'No'])

Is_active_member = st.selectbox('Active Member', ['Yes', 'No'])

# map numeric numbers to categorical variables
credit_card = 1 if Has_credit_card == 'Yes' else 0

Active_member = 1 if Is_active_member == 'Yes' else 0

# Predict button
if st.button('Predict'):
    # Prepare the input data
    input_data = pd.DataFrame({
        'CreditScore': [Credit_Score],
        'Gender': [label_encoder.transform([Gender])[0]],
        'Age': [Age],
        'Tenure': [Tenure],
        'Balance': [Balance],
        'NumOfProducts': [Number_of_products],
        'HasCrCard': [credit_card],
        'IsActiveMember': [Active_member],
        'EstimatedSalary': [Salary]
    })


    # Scale the input data
    columns_to_standardize = ['CreditScore','Age','Tenure','Balance','EstimatedSalary']
    input_data[columns_to_standardize] = scaler.transform(input_data[columns_to_standardize])

    # Predict churn
    pred = rf_model.predict(input_data)
    if pred[0] == 1:
        st.success("Customer is likely to Churn.")
    else:
        st.error("Customer will not Churn.")
