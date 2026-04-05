import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# Load model
model = pickle.load(open('model.pkl', 'rb'))

st.set_page_config(page_title="SmartGuard", layout="wide")

# Sidebar
st.sidebar.title("🔐 SmartGuard Dashboard")
page = st.sidebar.radio("Navigation", ["Home", "Single Prediction", "Bulk Prediction"])

# HOME
if page == "Home":
    st.title("💳 SmartGuard - Fraud Detection System")
    st.write("Machine Learning based system for detecting fraudulent transactions.")

# SINGLE PREDICTION
elif page == "Single Prediction":
    st.header("🔍 Enter Transaction Details")

    time = st.number_input("Time", 0.0)
    amount = st.number_input("Amount", 0.0)

    st.subheader("PCA Features (V1 - V28)")

    v_inputs = []
    for i in range(1, 29):
        val = st.number_input(f"V{i}", value=0.0)
        v_inputs.append(val)

    if st.button("Predict Fraud"):
        features = np.array([[time] + v_inputs + [amount]])

        prediction = model.predict(features)
        prob = model.predict_proba(features)

        if prediction[0] == 1:
            st.error(f"🚨 Fraud Detected! Probability: {prob[0][1]:.2f}")
        else:
            st.success(f"✅ Legitimate Transaction (Confidence: {1 - prob[0][1]:.2f})")

        # Probability chart
        fig, ax = plt.subplots()
        ax.bar(["Legit", "Fraud"], prob[0])
        st.pyplot(fig)

# BULK PREDICTION
elif page == "Bulk Prediction":
    st.header("📂 Upload CSV File")

    file = st.file_uploader("Upload CSV", type=["csv"])

    if file is not None:
    data = pd.read_csv(file)

    # ✅ Step 1: Remove 'Class' column if present
    if 'Class' in data.columns:
        data = data.drop('Class', axis=1)

    # ✅ Step 2: Fix column order (VERY IMPORTANT)
    expected_columns = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
    data = data[expected_columns]

    # ✅ Step 3: Make prediction
    predictions = model.predict(data)

    # ✅ Step 4: Show results
    data['Prediction'] = predictions

    st.write(data.head())

    fraud_count = data['Prediction'].sum()
    st.warning(f"🚨 Total Fraud Transactions: {fraud_count}")
