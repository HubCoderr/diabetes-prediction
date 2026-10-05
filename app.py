import streamlit as st
import numpy as np
import joblib

# Load trained model and scaler
model = joblib.load("diabetes_logistic_model.pkl")
scaler = joblib.load("diabetes_scaler.pkl")


# Page title
st.title("Diabetes Prediction App")

st.write(
    "Enter the patient's information below to predict "
    "the probability of Outcome = 1."
)


# User inputs
pregnancies = st.number_input(
    "Pregnancies",
    min_value=0,
    value=1
)

glucose = st.number_input(
    "Glucose",
    min_value=0.0,
    value=120.0
)

blood_pressure = st.number_input(
    "Blood Pressure",
    min_value=0.0,
    value=70.0
)

skin_thickness = st.number_input(
    "Skin Thickness",
    min_value=0.0,
    value=20.0
)

insulin = st.number_input(
    "Insulin",
    min_value=0.0,
    value=80.0
)

bmi = st.number_input(
    "BMI",
    min_value=0.0,
    value=25.0
)

diabetes_pedigree = st.number_input(
    "Diabetes Pedigree Function",
    min_value=0.0,
    value=0.5
)

age = st.number_input(
    "Age",
    min_value=1,
    value=30
)


# Prediction button
if st.button("Predict"):

    input_data = np.array([[
        pregnancies,
        glucose,
        blood_pressure,
        skin_thickness,
        insulin,
        bmi,
        diabetes_pedigree,
        age
    ]])

    # Scale the input using the same scaler used during training
    input_scaled = scaler.transform(input_data)

    # Prediction probability
    probability = model.predict_proba(input_scaled)[0][1]

    # Final prediction
    prediction = model.predict(input_scaled)[0]

    st.subheader("Prediction Result")

    st.write(
        f"Probability of Outcome = 1: {probability:.2%}"
    )

    if prediction == 1:
        st.error("Prediction: Outcome = 1")
    else:
        st.success("Prediction: Outcome = 0")