import streamlit as st
import pandas as pd
import joblib

# Load model, scaler and encoders
model = joblib.load("models/laptop_price_model.pkl")
scaler = joblib.load("models/scaler.pkl")
encoders = joblib.load("notebook/encoders.pkl")

st.set_page_config(page_title="Laptop Price Predictor", page_icon="💻")

st.title("💻 Laptop Price Prediction")
st.write("Select the laptop specifications and click Predict.")

# ---------------- Dropdown Inputs ----------------

company = st.selectbox(
    "Company",
    encoders["Company"].classes_
)

product = st.selectbox(
    "Product",
    encoders["Product"].classes_
)

typename = st.selectbox(
    "Laptop Type",
    encoders["TypeName"].classes_
)

screen = st.selectbox(
    "Screen Resolution",
    encoders["ScreenResolution"].classes_
)

cpu = st.selectbox(
    "Processor",
    encoders["Cpu"].classes_
)

memory = st.selectbox(
    "Storage",
    encoders["Memory"].classes_
)

gpu = st.selectbox(
    "Graphics Card",
    encoders["Gpu"].classes_
)

os = st.selectbox(
    "Operating System",
    encoders["OpSys"].classes_
)

# Numeric Inputs

inches = st.slider(
    "Screen Size (Inches)",
    10.0,
    20.0,
    15.6
)

ram = st.selectbox(
    "RAM (GB)",
    [2,4,8,12,16,24,32,64]
)

weight = st.number_input(
    "Weight (kg)",
    min_value=0.5,
    max_value=5.0,
    value=2.0,
    step=0.1
)

# ---------------- Prediction ----------------

if st.button("Predict Price"):

    input_df = pd.DataFrame({
        "laptop_ID":[0],
        "Company":[encoders["Company"].transform([company])[0]],
        "Product":[encoders["Product"].transform([product])[0]],
        "TypeName":[encoders["TypeName"].transform([typename])[0]],
        "Inches":[inches],
        "ScreenResolution":[encoders["ScreenResolution"].transform([screen])[0]],
        "Cpu":[encoders["Cpu"].transform([cpu])[0]],
        "Ram":[ram],
        "Memory":[encoders["Memory"].transform([memory])[0]],
        "Gpu":[encoders["Gpu"].transform([gpu])[0]],
        "OpSys":[encoders["OpSys"].transform([os])[0]],
        "Weight":[weight]
    })

    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)

    st.success(f"💰 Estimated Laptop Price: € {prediction[0]:,.2f}")