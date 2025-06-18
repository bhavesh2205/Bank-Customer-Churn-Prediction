from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle
import uvicorn

# Load the model, encoders and scaler
with open('pickle/random_forest.pkl', 'rb') as file:
    rf_model = pickle.load(file)

with open('pickle/label_encoder.pkl', 'rb') as file:
    label_encoder = pickle.load(file)

with open('pickle/scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Create FastAPI app
app = FastAPI(title="Customer Churn Prediction API", 
              description="API for predicting customer churn using machine learning models")

# Pydantic model for input validation
class CustomerData(BaseModel):
    gender: Literal["Male", "Female"] = Field(..., description="Customer gender")
    age: int = Field(..., ge=0, le=120, description="Customer age")
    balance: float = Field(..., ge=0, description="Account balance")
    credit_score: int = Field(..., ge=300, le=900, description="Credit score")
    salary: float = Field(..., ge=0, description="Estimated yearly salary")
    tenure: int = Field(..., ge=0, le=10, description="Number of years with the bank")
    number_of_products: int = Field(..., ge=1, le=4, description="Number of products used")
    has_credit_card: Literal["Yes", "No"] = Field(..., description="Whether customer has credit card")
    is_active_member: Literal["Yes", "No"] = Field(..., description="Whether customer is active member")

# Pydantic model for response
class PredictionResponse(BaseModel):
    prediction: bool
    message: str
    probability: float

@app.get("/health")
async def health_check():
    return {"message": "Customer Churn Prediction API", "status": "healthy", "model_loaded": True}


@app.post("/predict", response_model=PredictionResponse)
async def predict_churn(customer_data: CustomerData):
    try:
        # Map categorical variables to numeric
        credit_card = 1 if customer_data.has_credit_card == 'Yes' else 0
        active_member = 1 if customer_data.is_active_member == 'Yes' else 0
        
        # Prepare the input data
        input_data = pd.DataFrame({
            'CreditScore': [customer_data.credit_score],
            'Gender': [label_encoder.transform([customer_data.gender])[0]],
            'Age': [customer_data.age],
            'Tenure': [customer_data.tenure],
            'Balance': [customer_data.balance],
            'NumOfProducts': [customer_data.number_of_products],
            'HasCrCard': [credit_card],
            'IsActiveMember': [active_member],
            'EstimatedSalary': [customer_data.salary]
        })

        # Scale the input data
        columns_to_standardize = ['CreditScore', 'Age', 'Tenure', 'Balance', 'EstimatedSalary']
        input_data[columns_to_standardize] = scaler.transform(input_data[columns_to_standardize])

        # Predict churn
        pred = rf_model.predict(input_data)
        pred_proba = rf_model.predict_proba(input_data)[0]
        
        # Get prediction probability
        churn_probability = pred_proba[1] if pred[0] == 1 else pred_proba[0]
        
        # Create response
        if pred[0] == 1:
            message = "Customer is likely to Churn."
        else:
            message = "Customer will not Churn."
            
        return PredictionResponse(
            prediction=bool(pred[0]),
            message=message,
            probability=float(churn_probability)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
