from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

# Load trained model
model = joblib.load("models/churn_model.pkl")

# Input schema
class CustomerData(BaseModel):
    SeniorCitizen: int
    tenure: int
    MonthlyCharges: float
    TotalCharges: float

@app.get("/")
def home():
    return {
        "message": "Customer Churn Prediction API Running"
    }

@app.post("/predict")
def predict(data: CustomerData):

    input_data = pd.DataFrame([{
        "SeniorCitizen": data.SeniorCitizen,
        "tenure": data.tenure,
        "MonthlyCharges": data.MonthlyCharges,
        "TotalCharges": data.TotalCharges
    }])

    prediction = model.predict(input_data)

    result = "Customer May Churn"

    if prediction[0] == 0:
        result = "Customer Likely To Stay"

    return {
        "prediction": result
    }