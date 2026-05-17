from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

# Load saved model
model = joblib.load(r"C:\Users\Mounya\customer-churn-ltv\models\churn_model.pkl")

# Create FastAPI app
app = FastAPI()

# Input schema
class CustomerData(BaseModel):
    tenure: int
    MonthlyCharges: float
    TotalCharges: float
    SeniorCitizen: int

# Home route
@app.get("/")
def home():
    return {"message": "Customer Churn Prediction API Running"}

# Prediction route
@app.post("/predict")
def predict(data: CustomerData):

    # Convert input into dataframe
    input_data = pd.DataFrame([{
        "tenure": data.tenure,
        "MonthlyCharges": data.MonthlyCharges,
        "TotalCharges": data.TotalCharges,
        "SeniorCitizen": data.SeniorCitizen
    }])

    # Predict
    prediction = model.predict(input_data)[0]

    # Result
    if prediction == 1:
        result = "Customer Will Churn"
    else:
        result = "Customer Will Stay"

    return {"prediction": result}