# Import FastAPI to create the REST API
from fastapi import FastAPI

# Import BaseModel to validate incoming request data
from pydantic import BaseModel

# Import joblib to load the trained machine learning model
import joblib

# Import NumPy to create the input array for prediction
import numpy as np


# Create a FastAPI application instance
app = FastAPI()


# Load the trained diabetes prediction model
# The model was previously saved as diabetes_model.pkl
model = joblib.load("diabetes_model.pkl")


# Define the structure of the input data
# Pydantic automatically validates the data received through the API
class DiabetesInput(BaseModel):
    Pregnancies: int
    Glucose: float
    BloodPressure: float
    BMI: float
    Age: int


# Create a GET endpoint for the root URL
# This can be used to check whether the API is running
@app.get("/")
def read_root():
    return {"message": "Diabetes Prediction API is live"}


# Create a POST endpoint for making diabetes predictions
@app.post("/predict")
def predict(data: DiabetesInput):

    # Convert the input values into a NumPy array
    # The order must match the order of features used during model training
    input_data = np.array([[
        data.Pregnancies,
        data.Glucose,
        data.BloodPressure,
        data.BMI,
        data.Age
    ]])

    # Use the trained model to make a prediction
    # [0] extracts the first prediction from the returned array
    prediction = model.predict(input_data)[0]

    # Return the prediction as a JSON response
    # 0 = not diabetic
    # 1 = diabetic
    return {"diabetic": bool(prediction)}