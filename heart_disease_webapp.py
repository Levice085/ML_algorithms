import pandas as pd
import numpy as np
import streamlit as st
import requests
import pickle
loaded_model = pickle.load(open('C:/Users/levie/OneDrive/Desktop/Year 5/Data science/Python/models/heart_disease_model.sav','rb'))
def hd_prediction(X_train):
    X_train_np =np.asarray(X_train) 
    X_train_shaped = X_train_np.reshape(1,-1)
    y_pred =loaded_model.predict(X_train_shaped)
    print(y_pred)
    if y_pred[0]==0:
        return("The patient is has no heart disease")
    else:
        return("The patient has heart disease")
    
def main():
    #Giving a title
    st.title("Heart disease prediction Web app")

    age = st.text_input("Age of the patient")
    sex = st.text_input("The patient's gender:")
    Chest_Pain = st.text_input("Level of chest pain:")
    Blood_Pressure = st.text_inout("Blood pressure:")
    cholestoral = st.text_input("Level of cholestrol")
    Fasting_Blood_Sugar = st.text_input("Blood sugar levels:")
    resting_electrocardiographic = st.text_input("Resting:")
    Maximum_Heart_Rate = st.text_input("Max heartbeat:")
    Excersize_Includes = st.text_input("Excersize_Includes:")
    ST_Depression = st.text_input("Depression levels:")
    Slope_of_Excersize = st.text_input("Excersize:")
    Number_of_vessels = st.text_input("Number of vessels:")
    #Code for prediction
    diagnosis = ''
    #Creating a button for prediction
    if st.button("Heart disease test results:"):
        diagnosis = hd_prediction([age,sex,Chest_Pain,Blood_Pressure,cholestoral,Fasting_Blood_Sugar,
                                   resting_electrocardiographic,Maximum_Heart_Rate,Excersize_Includes,
                                   ST_Depression,Slope_of_Excersize,Number_of_vessels])
    st.success(diagnosis)
    if __name__=="__main__":
        main()


