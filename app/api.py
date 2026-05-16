from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Load trained model
model = joblib.load("models/fraud_model.pkl")

# Input schema
class Transaction(BaseModel):
    data: list

@app.get("/")
def home():
    return {"message": "Fraud Detection API Running"}

@app.post("/predict")
def predict(transaction: Transaction):

    data = np.array(transaction.data).reshape(1, -1)

    prediction = model.predict(data)[0]

    probability = model.predict_proba(data)[0][1]

    return {
        "prediction": int(prediction),
        "fraud_probability": float(probability)
    }