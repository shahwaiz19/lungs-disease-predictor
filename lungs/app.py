import streamlit as st
import joblib
import numpy as np
import os

# Load model and features
model = joblib.load("lungs_model.pkl")
features = joblib.load("lungs_model_features.pkl")

st.title("🫁 Lung Disease Prediction App")

st.subheader("Please fill all the details below:")
st.subheader("Developed by: Shahwaiz Amer")

# Manual fields based on your dataset
age = st.number_input("Age", min_value=0, max_value=120, step=1)
gender = st.selectbox("Gender", [0, 1])  # Assume 0 = Male, 1 = Female
air_pollution = st.slider("Air Pollution", 0, 8)
alcohol_use = st.slider("Alcohol Use", 0, 8)
dust_allergy = st.slider("Dust Allergy", 0, 8)
occupational_hazards = st.slider("Occupational Hazards", 0, 8)
genetic_risk = st.slider("Genetic Risk", 0, 7)
chronic_lung_disease = st.slider("Chronic Lung Disease", 0, 7)
balanced_diet = st.slider("Balanced Diet", 0, 8)
obesity = st.slider("Obesity", 0, 8)
smoking = st.slider("Smoking", 0, 8)
passive_smoker = st.slider("Passive Smoker", 0, 8)
chest_pain = st.slider("Chest Pain", 0, 8)
coughing_blood = st.slider("Coughing of Blood", 0, 8)
fatigue = st.slider("Fatigue", 0, 8)
weight_loss = st.slider("Weight Loss", 0, 8)
shortness_breath = st.slider("Shortness of Breath", 0, 8)
wheezing = st.slider("Wheezing", 0, 8)
swallowing_diff = st.slider("Swallowing Difficulty", 0, 8)
clubbing_nails = st.slider("Clubbing of Finger Nails", 0, 8)
frequent_cold = st.slider("Frequent Cold", 0, 8)
dry_cough = st.slider("Dry Cough", 0, 8)
snoring = st.slider("Snoring", 0, 8)

# Collect inputs in same order as model
input_data = np.array([[age, gender, air_pollution, alcohol_use, dust_allergy,
                        occupational_hazards, genetic_risk, chronic_lung_disease,
                        balanced_diet, obesity, smoking, passive_smoker,
                        chest_pain, coughing_blood, fatigue, weight_loss,
                        shortness_breath, wheezing, swallowing_diff,
                        clubbing_nails, frequent_cold, dry_cough, snoring]])

# Predict on button click
if st.button("Predict"):
    prediction = model.predict(input_data)
    levels = ['Low', 'Medium', 'High']
    st.success(f"Predicted Disease Level: {levels[prediction[0]]}")

st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .footer-hover {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        text-align: center;
        font-size: 14px;
        color: transparent;
        background-color: transparent;
        transition: all 0.3s ease;
        padding: 10px;
    }
    .footer-hover:hover {
        color: black;
        background-color: #f0f2f6;
    }
    </style>
    <div class="footer-hover">
        Developed by Shahwaiz | © 2025
    </div>
    """,
    unsafe_allow_html=True
)

