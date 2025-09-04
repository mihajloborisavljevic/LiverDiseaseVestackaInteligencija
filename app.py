import streamlit as st
import joblib
import numpy as np
from pathlib import Path

model_path = Path(__file__).parent / "notebook" / "models" / "best_liver_model.pkl"
model = joblib.load(model_path)

st.title("Predikcija bolesti jetre")

st.write("Unesi podatke pacijenta:")

age = st.number_input("Starost", min_value=0, max_value=120, value=30)
gender = st.selectbox("Pol", ["muški", "ženski"])
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
alcohol = st.number_input("Konzumacija alkohola (0-100)", min_value=0.0, max_value=100.0, value=10.0)
smoking = st.selectbox("Pušenje", [0, 1])
genetic_risk = st.number_input("Genetski rizik (0-1)", min_value=0.0, max_value=1.0, value=0.5)
physical_activity = st.number_input("Fizička aktivnost (0-1)", min_value=0.0, max_value=1.0, value=0.5)
diabetes = st.selectbox("Dijabetes", [0, 1])
hypertension = st.selectbox("Hipertenzija", [0, 1])
liver_test = st.number_input("Liver Function Test", min_value=0.0, max_value=100.0, value=40.0)

import pandas as pd

if st.button("Predikcija"):
    gender_val = 1 if gender == "muški" else 0
    input_data = pd.DataFrame([{
        "Age": age,
        "Gender": gender_val,
        "BMI": bmi,
        "AlcoholConsumption": alcohol,
        "Smoking": smoking,
        "GeneticRisk": genetic_risk,
        "PhysicalActivity": physical_activity,
        "Diabetes": diabetes,
        "Hypertension": hypertension,
        "LiverFunctionTest": liver_test
    }])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"Osoba ima bolest jetre (Verovatnoća: {probability:.2f})")
    else:
        st.success(f"Osoba nema bolest jetre (Verovatnoća: {1 - probability:.2f})")

