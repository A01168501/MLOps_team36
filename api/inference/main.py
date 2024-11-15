from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import joblib
import uvicorn

# Load the saved model
with open("Linear_Regression_degree_2.pkl", "rb") as f:
    model = joblib.load(f)

# Define the input data format for prediction
class AQData(BaseModel):
    features: List[float]

# Initialize FastAPI app
app = FastAPI()

# Define prediction endpoint
@app.post("/predict")
def predict(AQ_data: AQData):
    # Validate input length
    if len(AQ_data.features) != model.n_features_in_:
        raise HTTPException(
            status_code=400,
            detail=f"Input must contain {model.n_features_in_} features."
        )

    # Make prediction
    prediction = model.predict([AQ_data.features])[0]
    
    return {"prediction": int(prediction)}

# Define a root endpoint
@app.get("/")
def read_root():
    return {"message": "Air Quality Prediction model API"}

# Run the server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)