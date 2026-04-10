import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# ---------------- UI HEADER ----------------
st.set_page_config(page_title="EEG Seizure Forecasting", layout="wide")

st.title("🧠 Seizure Forecasting Dashboard")
st.markdown("### 📂 Drop your EEG file here to analyze brain activity and predict seizure risk")

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader("Upload EEG CSV File", type=["csv"])

if uploaded_file is not None:

    # ---------------- LOAD DATA ----------------
    df = pd.read_csv(uploaded_file)
    df = df.drop(columns=['Unnamed: 0'], errors='ignore')
    df = df.apply(pd.to_numeric, errors='coerce').fillna(0)

    # Extract signal
    signal = df.iloc[0, :-1].values.astype(float)

    # ---------------- DASHBOARD LAYOUT ----------------
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 EEG Signal")
        st.line_chart(signal)

    # ---------------- LSTM PROCESS ----------------
    scaler = MinMaxScaler()
    signal_scaled = scaler.fit_transform(signal.reshape(-1, 1))

    def create_sequences(data, seq_length=10):
        X = []
        for i in range(len(data) - seq_length):
            X.append(data[i:i+seq_length])
        return np.array(X)

    X = create_sequences(signal_scaled)

    # Load model (FIXED)
    model = load_model("lstm_model.h5", compile=False)

    predictions = model.predict(X)
    predictions = scaler.inverse_transform(predictions)

    with col2:
        st.subheader("📉 Predicted Signal (LSTM)")
        st.line_chart(predictions)

    # ---------------- FEATURE EXTRACTION ----------------
    mean_val = np.mean(signal)
    max_val = np.max(signal)
    std_val = np.std(signal)

    # ---------------- METRICS DISPLAY ----------------
    st.markdown("## 📊 Signal Metrics")

    m1, m2, m3 = st.columns(3)
    m1.metric("Mean", f"{mean_val:.2f}")
    m2.metric("Max Value", f"{max_val:.2f}")
    m3.metric("Standard Deviation", f"{std_val:.2f}")

    # ---------------- RISK ANALYSIS ----------------
    st.markdown("## 🧠 Seizure Risk Analysis")

    if max_val > 150:
        st.error("🚨 High Risk of Seizure")
        comment = "Abnormally high spikes detected in EEG signal. Immediate medical attention recommended."
        risk_level = "high"

    elif std_val > 40:
        st.warning("⚠️ Moderate Risk")
        comment = "High variability observed. Possible abnormal brain activity."
        risk_level = "moderate"

    else:
        st.success("✅ Normal EEG Activity")
        comment = "EEG signal appears stable with no major abnormalities."
        risk_level = "low"

    # ---------------- COMMENT SECTION ----------------
    st.markdown("## 📝 System Interpretation")
    st.info(comment)

    # ---------------- INTERACTIVE INPUT ----------------
    if risk_level in ["moderate", "high"]:

        st.markdown("## 🧾 Patient Information")

        age = st.number_input("Age", 1, 100)
        problem = st.selectbox(
            "Select Primary Issue",
            ["Frequent Headache", "Previous Seizures", "Memory Loss", "No prior issues"]
        )

        medication = st.selectbox(
            "Current Medication",
            ["None", "Anti-epileptic drugs", "Painkillers", "Other"]
        )

        st.markdown("## 💊 AI Recommendation")

        if risk_level == "high":
            st.error("🚨 Immediate neurologist consultation required")

            if medication == "None":
                st.warning("⚠️ Patient is not on medication. Clinical evaluation needed urgently.")

            elif medication == "Anti-epileptic drugs":
                st.info("🔍 Monitor drug effectiveness. EEG shows abnormal spikes.")

            else:
                st.info("⚠️ Current medication may not be sufficient. Doctor review required.")

        elif risk_level == "moderate":
            st.warning("⚠️ Regular monitoring recommended")

            if problem == "Previous Seizures":
                st.info("🔍 Possible recurrence risk. Continue monitoring.")

            else:
                st.info("📊 Mild abnormality detected. Lifestyle and sleep monitoring advised.")