from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

# Load trained model
model = joblib.load("models/churn_model.pkl")

@app.get("/")
def home():
    return {"message": "Customer Churn Prediction API Running"}

@app.post("/predict")
def predict():

    sample_data = {
        "SeniorCitizen": 0,
        "tenure": 12,
        "MonthlyCharges": 70,
        "TotalCharges": 850
    }

    input_data = pd.DataFrame([sample_data])

    prediction = model.predict(input_data)

    return {
        "prediction": int(prediction[0])
    }