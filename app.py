import streamlit as st
import numpy as np
import joblib

st.title("❤️ Heart Disease Prediction App")

# Load model
model = joblib.load("heart_model.pkl")

st.subheader("Enter Patient Details")

age = st.number_input("Age", 20, 100, 50)
sex = st.selectbox("Sex (1 = Male, 0 = Female)", [1, 0])
cp = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
chol = st.number_input("Cholesterol", 100, 400, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (1 = Yes, 0 = No)", [1, 0])
restecg = st.selectbox("Resting ECG Results (0-2)", [0, 1, 2])
thalach = st.number_input("Max Heart Rate Achieved", 70, 220, 150)
exang = st.selectbox("Exercise Induced Angina (1 = Yes, 0 = No)", [1, 0])
oldpeak = st.number_input("ST Depression Induced by Exercise", 0.0, 6.0, 1.0)
slope = st.selectbox("Slope of Peak Exercise ST Segment (0-2)", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels (0-3)", [0, 1, 2, 3])
thal = st.selectbox("Thalassemia (1=Normal, 2=Fixed Defect, 3=Reversible Defect)", [1, 2, 3])

features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                      thalach, exang, oldpeak, slope, ca, thal]])

if st.button("Predict"):
    prediction = model.predict(features)
    if prediction[0] == 1:
        st.error("⚠️ Patient likely has Heart Disease")
    else:
        st.success("✅ Patient likely does NOT have Heart Disease")
