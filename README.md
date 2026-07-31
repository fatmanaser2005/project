# Meal Demand Prediction

## Overview
This project aims to predict the number of meal orders using machine learning regression models. The solution is built around a complete data science pipeline that includes data loading, merging, preprocessing, feature engineering, model training, evaluation, and model persistence.

## Project Goal
The main objective is to forecast meal demand accurately based on historical information such as meal characteristics, fulfillment center details, pricing, and discounts. Accurate demand prediction can support better planning, inventory management, and operational efficiency.

## Dataset
The project uses multiple CSV files that contain information about:
- meal details
- fulfillment center information
- pricing and discount values
- target variable: number of orders

The data is merged and processed before model training.

## Data Preprocessing
The preprocessing pipeline includes:
- removing duplicate rows
- handling missing values for both categorical and numerical columns
- creating new features such as:
  - discount
  - discount_percentage
- encoding categorical variables using LabelEncoder
- scaling numerical features using StandardScaler
- splitting the data into training and validation sets

## Feature Engineering
New derived variables were added to improve model performance:
- discount = base_price - checkout_price
- discount_percentage = (discount / base_price) * 100

These engineered features help capture the effect of pricing changes on demand.

## Models Trained
Several regression models were trained and evaluated:
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor

Each model was trained on the preprocessed dataset and saved as a serialized model file in the models folder.

## Evaluation Metrics
The models were evaluated using:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- MAPE (Mean Absolute Percentage Error)
- R2 Score

## Results
The current project reports the following sample results from the evaluation phase:
- MAE: 69.70
- RMSE: 147.59
- R2 Score: 0.857

These values indicate that the model provides a strong prediction performance for the given dataset.

## Project Structure
- data/: contains raw and processed data files
- src/data/: scripts for loading, merging, and preprocessing data
- src/models/: training, evaluation, and prediction scripts for different models
- models/: saved trained models and preprocessing artifacts
- notebooks/: notebooks used for data understanding and EDA
- reports/: EDA reports and generated figures

## How to Run
1. Install dependencies:
   pip install -r requirements.txt
2. Run the training pipeline:
   python src/models/train.py
3. Use the trained model for prediction through the prediction scripts in the src/models directory.

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Jupyter Notebook

## Notes
This repository contains both the experimental notebooks and the reusable Python scripts for the full machine learning workflow.
