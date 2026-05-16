from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

# Load trained model
model = joblib.load("models/churn_model.pkl")


@app.get("/")
def home():
    return {
        "message": "Customer Churn Prediction API Running"
    }


@app.get("/predict")
def predict():

    sample_data = pd.DataFrame([{
        "SeniorCitizen": 1,
        "tenure": 5,
        "MonthlyCharges": 95,
        "TotalCharges": 450
    }])

    prediction = model.predict(sample_data)

    result = "Customer May Churn"

    if prediction[0] == 0:
        result = "Customer Likely To Stay"

    return {
        "prediction": result
    }