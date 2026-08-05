import streamlit as st
import pandas as pd
import joblib

# Load Model and Scaler
model = joblib.load("models/best_model.pkl")
scaler = joblib.load("models/scaler.pkl")

st.set_page_config(
    page_title="Mobile Price Classification",
    page_icon="📱",
    layout="wide"
)

st.title("📱 Mobile Price Classification System")
st.markdown("### Predict the price category of a mobile phone")

st.sidebar.header("Enter Mobile Specifications")

# Inputs
battery_power = st.sidebar.number_input("Battery Power", 500, 2000, 1200)

blue = st.sidebar.selectbox("Bluetooth", [0, 1])

clock_speed = st.sidebar.slider("Clock Speed", 0.5, 3.0, 2.0)

dual_sim = st.sidebar.selectbox("Dual SIM", [0, 1])

fc = st.sidebar.number_input("Front Camera (MP)", 0, 20, 5)

four_g = st.sidebar.selectbox("4G", [0, 1])

int_memory = st.sidebar.number_input("Internal Memory (GB)", 2, 128, 32)

m_dep = st.sidebar.slider("Mobile Depth", 0.1, 1.0, 0.5)

mobile_wt = st.sidebar.number_input("Weight (g)", 80, 250, 150)

n_cores = st.sidebar.slider("Processor Cores", 1, 8, 4)

pc = st.sidebar.number_input("Primary Camera (MP)", 0, 25, 12)

px_height = st.sidebar.number_input("Pixel Height", 0, 2000, 800)

px_width = st.sidebar.number_input("Pixel Width", 500, 2500, 1200)

ram = st.sidebar.number_input("RAM (MB)", 256, 4000, 2000)

sc_h = st.sidebar.number_input("Screen Height", 5, 20, 12)

sc_w = st.sidebar.number_input("Screen Width", 1, 20, 6)

talk_time = st.sidebar.number_input("Talk Time (Hours)", 2, 30, 10)

three_g = st.sidebar.selectbox("3G", [0, 1])

touch_screen = st.sidebar.selectbox("Touch Screen", [0, 1])

wifi = st.sidebar.selectbox("WiFi", [0, 1])

# Prediction Button
if st.button("Predict Price Category"):

    data = pd.DataFrame([[
        battery_power,
        blue,
        clock_speed,
        dual_sim,
        fc,
        four_g,
        int_memory,
        m_dep,
        mobile_wt,
        n_cores,
        pc,
        px_height,
        px_width,
        ram,
        sc_h,
        sc_w,
        talk_time,
        three_g,
        touch_screen,
        wifi
    ]], columns=[
        'battery_power',
        'blue',
        'clock_speed',
        'dual_sim',
        'fc',
        'four_g',
        'int_memory',
        'm_dep',
        'mobile_wt',
        'n_cores',
        'pc',
        'px_height',
        'px_width',
        'ram',
        'sc_h',
        'sc_w',
        'talk_time',
        'three_g',
        'touch_screen',
        'wifi'
    ])

    data_scaled = scaler.transform(data)

    prediction = model.predict(data_scaled)[0]

    categories = {
        0: "🟢 Low Cost",
        1: "🟡 Medium Cost",
        2: "🟠 High Cost",
        3: "🔴 Very High Cost"
    }

    st.success(f"Predicted Price Category: **{categories[prediction]}**")

st.markdown("---")
st.caption("Developed using Machine Learning and Streamlit")