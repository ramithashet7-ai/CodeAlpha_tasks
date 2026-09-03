import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("car_price_model.pkl")

# Page configuration
st.set_page_config(
    page_title="Car Selling Price Predictor",
    page_icon="🚗",
    layout="centered"
)

st.title("🚗 Car Selling Price Predictor")
st.write("Enter the car details below to estimate its selling price.")

# User inputs
present_price = st.number_input(
    "Present Price (₹ lakhs)",
    min_value=0.1,
    value=5.0,
    step=0.1
)

driven_kms = st.number_input(
    "Kilometers Driven",
    min_value=0,
    value=30000,
    step=1000
)

fuel_type = st.selectbox(
    "Fuel Type",
    ["Petrol", "Diesel", "CNG"]
)

selling_type = st.selectbox(
    "Selling Type",
    ["Dealer", "Individual"]
)

transmission = st.selectbox(
    "Transmission",
    ["Manual", "Automatic"]
)

owner = st.selectbox(
    "Number of Previous Owners",
    [0, 1, 2, 3]
)

car_age = st.number_input(
    "Car Age (years)",
    min_value=0,
    max_value=30,
    value=4,
    step=1
)

# Prediction
if st.button("Predict Selling Price"):

    input_data = pd.DataFrame({
        "Present_Price": [present_price],
        "Driven_kms": [driven_kms],
        "Fuel_Type": [fuel_type],
        "Selling_type": [selling_type],
        "Transmission": [transmission],
        "Owner": [owner],
        "Car_Age": [car_age]
    })

    prediction = model.predict(input_data)[0]

    st.success(
        f"Estimated Selling Price: ₹{prediction:.2f} lakhs"
    )