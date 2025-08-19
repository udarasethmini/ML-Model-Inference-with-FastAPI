from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
from typing import List
import sklearn as st
import json

# Load the trained model and metadata on startup
model = joblib.load("model.pkl")
with open("model_info.json", "r") as f:
    model_info = json.load(f)
class_names = model_info["classes"]

app = FastAPI(title="Iris Classification API", description="API for Iris flower species classification using a trained ML model")

# Define input schema for a single prediction (list of 4 floats)
class PredictionInput(BaseModel):
    features: List[float]

    class Config:
        schema_extra = {
            "example": {
                "features": [5.1, 3.5, 1.4, 0.2]
            }
        }

# Define output schema
class PredictionOutput(BaseModel):
    prediction: str
    confidence: float  # Max probability as confidence
    probabilities: dict  # Full probabilities for each class

@app.get("/")
def health_check():
    return {"status": "healthy", "message": "Iris Classification API is running"}

@app.post("/predict", response_model=PredictionOutput)
def predict(input_data: PredictionInput):
    try:
        if len(input_data.features) != 4:
            raise ValueError("Input must have exactly 4 features: sepal length, sepal width, petal length, petal width")
        
        # Convert input to numpy array for model prediction
        features = np.array([input_data.features])
        
        # Make prediction
        pred_idx = model.predict(features)[0]
        proba = model.predict_proba(features)[0]
        
        # Get class name and probabilities
        prediction = class_names[pred_idx]
        probabilities = {class_names[i]: float(proba[i]) for i in range(len(proba))}
        confidence = float(max(proba))  # Use max probability as confidence
        
        return PredictionOutput(
            prediction=prediction,
            confidence=confidence,
            probabilities=probabilities
        )
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/model-info")
def get_model_info():
    return {
        "model_type": model_info["model_type"],
        "problem_type": model_info["problem_type"],
        "created_at": model_info["created_at"],
        "features": model_info["features"],
        "classes": model_info["classes"],
        "metrics": model_info["metrics"]
    }