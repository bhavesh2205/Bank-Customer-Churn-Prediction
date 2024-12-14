import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# Load the trained model
model = tf.keras.models.load_model('pickle/model_ANN.h5')
#with open('pickle/random_forest.pkl', 'rb') as file:
#    rf_model = pickle.load(file)

# Load the encoders and scaler
with open('pickle/label_encoder.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('pickle/onehot_encoder.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('pickle/scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Streamlit app
st.set_page_config(page_title='Customer Churn Prediction', layout='centered', initial_sidebar_state='expanded')
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 4px;
    }
    </style>
""", unsafe_allow_html=True)

st.title('Customer Churn Prediction')
st.markdown("""<h5 style='text-align: center; color: black;'>Predict whether a customer will churn based on their bank information</h5>""", unsafe_allow_html=True)

# User input
st.sidebar.header('Fill Below Details')
Geography = st.sidebar.selectbox('Geography', onehot_encoder_geo.categories_[0])
Gender = st.sidebar.selectbox('Gender', label_encoder_gender.classes_)
Age = st.sidebar.slider('Age', 18, 92)
Balance = st.sidebar.number_input('Balance', min_value=0.0, step=0.01)
Credit_Score = st.sidebar.slider('Credit Score', 300, 900)
Salary = st.sidebar.number_input('Salary', min_value=0.0, step=0.01)
Tenure = st.sidebar.slider('Tenure', 0, 10)
Number_of_products = st.sidebar.slider('Number of Products', 1, 4)
Has_credit_card = st.sidebar.selectbox('Has Credit Card', ['Yes', 'No'])
Is_active_member = st.sidebar.selectbox('Is Active Member', ['Yes', 'No'])

# Map Yes/No to 1/0
Has_credit_card = 1 if Has_credit_card == 'Yes' else 0
Is_active_member = 1 if Is_active_member == 'Yes' else 0

# Predict button
if st.sidebar.button('Predict'):
    # Prepare the input data
    input_data = pd.DataFrame({
        'CreditScore': [Credit_Score],
        'Gender': [label_encoder_gender.transform([Gender])[0]],
        'Age': [Age],
        'Tenure': [Tenure],
        'Balance': [Balance],
        'NumOfProducts': [Number_of_products],
        'HasCrCard': [Has_credit_card],
        'IsActiveMember': [Is_active_member],
        'EstimatedSalary': [Salary]
    })

    # One-hot encode 'Geography'
    geography_encoded = onehot_encoder_geo.transform([[Geography]]).toarray()
    geography_encoded_df = pd.DataFrame(geography_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

    # Combine one-hot encoded columns with input data
    input_data = pd.concat([input_data.reset_index(drop=True), geography_encoded_df], axis=1)

    # Scale the input data
    columns_to_standardize = ['CreditScore','Age','Tenure','Balance','EstimatedSalary']
    input_data[columns_to_standardize] = scaler.transform(input_data[columns_to_standardize])

    # Predict churn
    prediction = model.predict(input_data)
    prediction_proba = prediction[0][0]

    st.markdown("""<h4 style='text-align: center;'>Prediction Results</h4>""", unsafe_allow_html=True)

    st.write(f'**Churn Probability:** {prediction_proba:.2f}')

    if prediction_proba > 0.5:
        st.write("""<h5 style='color: red; text-align: center;'>The customer is likely to churn.</h5>""", unsafe_allow_html=True)
    else:
        st.write("""<h5 style='color: green; text-align: center;'>The customer is not likely to churn.</h5>""", unsafe_allow_html=True)

# Footer
st.markdown("""<footer style='position: fixed; bottom: 0; width: 100%; text-align: center; color: grey;'>Made with ❤️ using Streamlit</footer>""", unsafe_allow_html=True)
