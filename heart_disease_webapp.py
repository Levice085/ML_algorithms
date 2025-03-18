import pandas as pd
import numpy as np
import streamlit as st
import requests
import pickle
import os

# Downloading the model file from GitHub
url = "https://github.com/Levice085/ML_model_files/raw/main/heart_disease_model.sav"

loaded_model = requests.get(url)

# Save the downloaded content to a temporary file
with open('trained_model1.sav', 'wb') as f:
    pickle.dump(loaded_model, f)


# Load the saved model
with open('trained_model1.sav', 'rb') as f:
    loaded_model = pickle.load(f)


# Function to make predictions
def hd_prediction(X_train):
    X_train_np = np.asarray(X_train, dtype=float)  # Convert input to float
    X_train_shaped = X_train_np.reshape(1, -1)
    y_pred = loaded_model.predict(X_train_shaped)
    
    if y_pred[0] == 0:
        return "The patient has no heart disease"
    else:
        return "The patient has heart disease"

# Streamlit App
def main():
    # Title
    st.title("Heart Disease Prediction Web App")

    # User Inputs
    age = st.number_input("Age of the patient", min_value=1, max_value=120, step=1)
    sex = st.radio("The patient's gender:", options=[0, 1])  # 0 = Female, 1 = Male
    Chest_Pain = st.number_input("Level of chest pain", min_value=0, max_value=3, step=1)
    Blood_Pressure = st.number_input("Blood pressure", min_value=80, max_value=200, step=1)
    cholestoral = st.number_input("Level of cholesterol", min_value=100, max_value=500, step=1)
    Fasting_Blood_Sugar = st.radio("Fasting Blood Sugar > 120 mg/dl", options=[0, 1])
    resting_electrocardiographic = st.number_input("Resting ECG results", min_value=0, max_value=2, step=1)
    Maximum_Heart_Rate = st.number_input("Max Heart Rate", min_value=60, max_value=220, step=1)
    Excersize_Includes = st.radio("Exercise Induced Angina", options=[0, 1])
    ST_Depression = st.number_input("ST Depression", min_value=0.0, max_value=6.2, step=0.1)
    Slope_of_Excersize = st.number_input("Slope of Exercise", min_value=0, max_value=2, step=1)
    Number_of_vessels = st.number_input("Number of Major Vessels", min_value=0, max_value=4, step=1)

    # Prediction
    diagnosis = ''
    if st.button("Get Heart Disease Prediction"):
        diagnosis = hd_prediction([
            age, sex, Chest_Pain, Blood_Pressure, cholestoral, Fasting_Blood_Sugar,
            resting_electrocardiographic, Maximum_Heart_Rate, Excersize_Includes,
            ST_Depression, Slope_of_Excersize, Number_of_vessels
        ])
    
    # Display Result
    st.success(diagnosis)

# Run the app
if __name__ == "__main__":
    main()
