import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="EEG Intelligence Dashboard",
    layout="wide",
    page_icon="🧠"
)

# ---------------- HEADER ----------------
st.title("🧠 EEG Intelligence System")
st.markdown("### Seizure Prediction + EEG Wave Forecasting Dashboard")

st.divider()

# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_models():
    cnn = load_model("cnn_model.h5", compile=False)
    lstm = load_model("lstm_model.h5", compile=False)
    return cnn, lstm

cnn_model, lstm_model = load_models()

# ---------------- UPLOAD ----------------
uploaded_file = st.file_uploader("📂 Upload EEG CSV File", type=["csv"])

# ======================================================
# MAIN LOGIC
# ======================================================
if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)
    df = df.drop(columns=['Unnamed: 0'], errors='ignore')
    df = df.apply(pd.to_numeric, errors='coerce').fillna(0)

    signal = df.iloc[0, :-1].values.astype(float)

    # ---------------- SIDEBAR ----------------
    st.sidebar.title("⚙️ Analysis Panel")
    st.sidebar.write("Signal length:", len(signal))
    st.sidebar.write("Models loaded: CNN + LSTM")

    # ---------------- SCALING ----------------
    scaler = MinMaxScaler()
    signal_scaled = scaler.fit_transform(signal.reshape(-1, 1))

    # ---------------- LSTM SEQUENCES ----------------
    def create_sequences(data, seq_length=10):
        X = []
        for i in range(len(data) - seq_length):
            X.append(data[i:i+seq_length])
        return np.array(X)

    X_lstm = create_sequences(signal_scaled)

    if len(X_lstm) == 0:
        st.error("❌ Signal too short for LSTM prediction")
        st.stop()

    # ---------------- LSTM PREDICTION ----------------
    predictions = lstm_model.predict(X_lstm)
    predictions = scaler.inverse_transform(predictions)

    # ---------------- CNN PREDICTION (FIXED) ----------------
    cnn_input = signal.reshape(1, 178, 1)

    cnn_pred = cnn_model.predict(cnn_input)[0]

    prob_normal = float(cnn_pred[0])
    prob_seizure = float(cnn_pred[1])

    result = np.argmax(cnn_pred)

    # 🔥 FIXED RISK SCORE (NO FALSE HIGH RISK ISSUE)
    risk_score = prob_seizure

    # ---------------- METRICS ----------------
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("📊 Mean Signal", f"{np.mean(signal):.2f}")

    with col2:
        st.metric("📈 Max Signal", f"{np.max(signal):.2f}")

    with col3:
        st.metric("📉 Std Deviation", f"{np.std(signal):.2f}")

    st.divider()

    # ---------------- GRAPHS ----------------
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Original EEG Signal")
        st.line_chart(signal)

    with col2:
        st.subheader("🔮 Predicted EEG Signal (LSTM)")
        st.line_chart(predictions)

    st.divider()

    # ---------------- SEIZURE RISK (FIXED LOGIC) ----------------
    st.subheader("🚨 Seizure Risk Analysis")

    # 🔥 IMPROVED SCORING (NO MORE ALWAYS HIGH RISK)
    if prob_seizure >= 0.85 and prob_seizure > prob_normal:
        st.error(f"🔥 VERY HIGH SEIZURE RISK ({prob_seizure:.2f})")

    elif prob_seizure >= 0.65:
        st.warning(f"⚠️ HIGH SEIZURE RISK ({prob_seizure:.2f})")

    elif prob_seizure >= 0.40:
        st.info(f"🟡 MODERATE RISK ({prob_seizure:.2f})")

    else:
        st.success(f"✅ LOW RISK ({prob_seizure:.2f})")

    st.divider()

    # ---------------- BRAIN ANALYSIS ----------------
    st.subheader("🧠 Brain Activity Interpretation")

    signal_var = np.var(signal)
    pred_var = np.var(predictions)
    overall_var = (signal_var + pred_var) / 2

    if prob_seizure >= 0.85:
        st.error("Seizure activity strongly detected by CNN model.")

    elif prob_seizure >= 0.65:
        st.warning("High seizure probability detected → monitor closely.")

    elif overall_var > np.percentile(signal_var, 90):
        st.info("Mild abnormal activity detected.")

    else:
        st.success("Normal brain activity detected.")

else:
    st.info("👆 Upload EEG CSV file to start analysis")