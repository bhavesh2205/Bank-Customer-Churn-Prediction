# Bank Customer Churn Prediction Model

## Problem Statement

Customer churn is a significant challenge for banks and financial institutions. Churn occurs when customers stop using a bank's services, leading to a loss of revenue and potentially damaging the bank's reputation. Understanding and predicting customer churn is crucial for banks to take proactive measures to retain customers and enhance their services.

The objective of this project is to develop a predictive model that can identify customers who are likely to churn. By analyzing various features such as customer demographics, account information, and transaction history, we aim to gain insights into the factors contributing to churn and develop strategies to mitigate it.

## Features of the Project
- **EDA**: Provides detailed insights into the dataset through statistical summaries and visualizations.
- **Standardization**: Applies preprocessing selectively to specific columns for better model performance.
- **Multiple Models**: Trains and evaluates logistic regression, decision trees, random forest, SVM, KNN, and boosting algorithms (XGBoost, LightGBM, CatBoost).
- **Metrics Tracking**: Captures key performance metrics (Accuracy, Precision, Recall, F1-Score, and ROC-AUC) for each model.
- **Visualization**: Plots performance comparisons for easy analysis.


## Dataset Description
The dataset contains the following columns:
- **CustomerID**: Unique identifier for customers.
- **Gender**: Gender of the customer.
- **Age**: Customer's age.
- **CreditScore**: Creditworthiness score.
- **Geography**: Customer's country.
- **Tenure**: Number of years the customer has been with the bank.
- **Balance**: Account balance of the customer.
- **NumOfProducts**: Number of products the customer uses.
- **HasCrCard**: Whether the customer owns a credit card (1 = Yes, 0 = No).
- **IsActiveMember**: Whether the customer is an active member (1 = Yes, 0 = No).
- **EstimatedSalary**: Estimated yearly salary of the customer.
- **Exited**: Target variable indicating if a customer churned (1) or not (0).


## Data Exploration and Cleaning

  - Analyze missing values, outliers, and data distribution.
  - Applied onehot Encoder and label encoder for categorical variables (e.g., geography, gender).
  - Remove irrelevent columns from the dataset.

## Feature Engineering

  - Scale numerical features to improve model performance.
  - Select important features using correlation analysis and feature importance.

## Model Training

  - Split data into training and testing sets (80-20 split).
  - Since the data was imbalanced so handled this issue using SMOTE (Synthetic Minority Over-sampling Technique) in order to make fair predictions.
  - Created a function for training different models in one go like Logistic Regression, Random Forest, SVC, Decision Tree, XGBoost, CatBoot etc.
  - Also trained ANN model using tensorflow framework.

## Evaluation

  - Evaluated model performance using metrics like accuracy, precision, recall, F1-score, and ROC-AUC.
  - Ramdom forest model found out to be the best performing model with recall score of 89%.
    
## Deployment

  - Save the trained model using pickle.
  - Deployed web app via streamlit open source community cloud.





