import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set page config
st.set_page_config(page_title="Body Performance Prediction", layout="centered")

st.title("🏋️‍♂️ Body Performance Predictor")

with st.sidebar:
    st.header("ℹ️ About")
    st.info("This application uses a trained **Neural Network (MLP)** to predict your body performance class based on standard physical measurements.")
    st.markdown("---")
    st.markdown("**Classes Definition:**\n- **A:** Best Performance\n- **B:** Good Performance\n- **C:** Average Performance\n- **D:** Poor Performance")

st.write("Enter your physical measurements below to predict your performance class.")

@st.cache_resource
def load_models():
    import os
    model_path = os.path.join('models', 'model.joblib')
    scaler_path = os.path.join('models', 'scaler.joblib')
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

try:
    model, scaler = load_models()
except Exception as e:
    st.error("Model files not found. Please train the model first by running `python train.py`.")
    st.stop()

# Inputs
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age (years)", min_value=15, max_value=80, value=25)
    gender = st.selectbox("Gender", options=["Male", "Female"])
    height = st.number_input("Height (cm)", min_value=100.0, max_value=220.0, value=170.0)
    weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=70.0)
    body_fat = st.number_input("Body Fat (%)", min_value=2.0, max_value=60.0, value=20.0)

with col2:
    diastolic = st.number_input("Diastolic BP (mmHg)", min_value=40.0, max_value=120.0, value=80.0)
    systolic = st.number_input("Systolic BP (mmHg)", min_value=70.0, max_value=200.0, value=120.0)
    grip = st.number_input("Grip Strength (kg)", min_value=0.0, max_value=100.0, value=40.0)
    flexibility = st.number_input("Flexibility - Sit and Bend Forward (cm)", min_value=-30.0, max_value=40.0, value=15.0)
    situps = st.number_input("Sit-ups count", min_value=0, max_value=100, value=40)

# Process inputs
gender_enc = 1 if gender == "Male" else 0
features = np.array([[age, gender_enc, height, weight, body_fat, diastolic, systolic, grip, flexibility, situps]])

if st.button("Predict Performance Class", type="primary"):
    # Scale features
    features_scaled = scaler.transform(features)
    
    # Predict
    pred_idx = model.predict(features_scaled)[0]
    prob = model.predict_proba(features_scaled)[0]
    
    class_mapping = {0: 'A (Best)', 1: 'B (Good)', 2: 'C (Average)', 3: 'D (Poor)'}
    pred_class = class_mapping[pred_idx]
    
    st.markdown("---")
    st.subheader(f"Predicted Class: **{pred_class}**")
    
    # Show probabilities
    st.write("Prediction Probabilities:")
    prob_df = pd.DataFrame({
        "Class": ["A", "B", "C", "D"],
        "Probability": prob
    }).set_index("Class")
    st.bar_chart(prob_df)
