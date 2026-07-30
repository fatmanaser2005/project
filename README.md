# Demand Forecasting Project

## Overview
This project builds a machine learning model to forecast product demand using historical order data and supporting features such as price, promotions, meal category, cuisine, center information, and operational area.

The goal is to predict the target variable `num_orders` for future weeks using a baseline regression pipeline.

## Problem Statement
Food demand can vary significantly by week, product type, region, and marketing activity. Forecasting demand accurately helps improve inventory planning, reduce waste, and support better operational decisions.

## Dataset
The project uses historical data from:
- `data/raw/train.csv`
- `data/raw/meal_info.csv`
- `data/raw/fulfilment_center_info.csv`

### Dataset Summary
- Rows: 456,548
- Features: 14 input features + target column
- Target variable: `num_orders`
- Missing values: none after preprocessing

## Project Structure
```text
project/
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── notebooks/
├── reports/
├── src/
│   ├── data/
│   │   ├── load_data.py
│   │   ├── merge_data.py
│   │   └── preprocess.py
│   └── models/
│       ├── evaluate.py
│       ├── predict.py
│       └── train.py
└── requirements.txt
```

## Installation
1. Create and activate a virtual environment
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```
2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

## How to Run
### Train the model
```bash
python src/models/train.py
```

### Generate predictions
```bash
python src/models/predict.py
```

## Machine Learning Pipeline
1. Load raw data from the CSV files.
2. Merge meal and fulfillment center information with training data.
3. Preprocess the dataset:
   - remove duplicates
   - fill missing values
   - engineer discount-related features
   - encode categorical variables
   - scale numeric features
4. Split data into training and validation sets.
5. Train a `RandomForestRegressor`.
6. Evaluate the model using MAE, RMSE, MAPE, and R².
7. Save the trained model and preprocessing artifacts.

## Model
The current baseline model uses:
- `RandomForestRegressor`
- `train_test_split` with a validation split
- preprocessing steps including feature engineering and scaling

The trained model is stored in:
- `models/best_model.pkl`
- `models/scaler.pkl`
- `models/encoders.pkl`
- `models/evaluation_metrics.json`

## Results
The current model performance on the validation set is:
- MAE: 69.70
- RMSE: 147.60
- MAPE: 49.84%
- R² Score: 0.8572

These results are strong for a first baseline model and indicate that the model captures a large portion of demand variability.

## Notes
The project currently uses a baseline approach. Future improvements could include:
- lag features and rolling averages
- stronger time-based features
- more advanced models such as gradient boosting
- target transformation for highly skewed demand values