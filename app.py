import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import shap

# ==============================
# Load Model
# ==============================
model = pickle.load(open('model.pkl', 'rb'))

# SHAP Explainer
explainer = shap.Explainer(model)

st.set_page_config(page_title="SmartGuard", layout="wide")

# ==============================
# Helper Functions
# ==============================
def get_risk_level(prob):
    if prob > 0.8:
        return "🔴 High Risk"
    elif prob > 0.4:
        return "🟡 Medium Risk"
    else:
        return "🟢 Low Risk"

# ==============================
# Sidebar
# ==============================
st.sidebar.title("🔐 SmartGuard Dashboard")
page = st.sidebar.radio("Navigation", ["Home", "Single Prediction", "Bulk Prediction"])

# ==============================
# HOME
# ==============================
if page == "Home":
    st.title("💳 SmartGuard - Fraud Detection System")


# ==============================
# SINGLE PREDICTION
# ==============================
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
        prob = model.predict_proba(features)[0][1]

        risk = get_risk_level(prob)

        # Result
        if prediction[0] == 1:
            st.error(f"🚨 Fraud Detected! Probability: {prob:.2f}")
        else:
            st.success(f"✅ Legitimate Transaction (Confidence: {1 - prob:.2f})")

        st.info(f"Risk Level: {risk}")

        # Probability Chart
        fig, ax = plt.subplots()
        ax.bar(["Legit", "Fraud"], [1 - prob, prob])
        st.pyplot(fig)

        # ==============================
        # 🔥 XAI - SHAP Explanation
        # ==============================
        
        st.subheader("🧠 Explainable AI (Why this prediction?)")

        try:
          feature_names = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
          input_df = pd.DataFrame(features, columns=feature_names)

          shap_values = explainer(input_df)

          # ✅ Fix for classification models
          if hasattr(shap_values, "values"):
              values = shap_values.values
          else:
              values = shap_values

          # If multi-class → pick fraud class (index 1)
          if len(values.shape) == 3:
              values = values[:, :, 1]

          # Plot manually (SAFE)
          fig, ax = plt.subplots()
          ax.barh(feature_names, values[0])
          ax.set_title("Feature Contribution")
          st.pyplot(fig)

     except Exception as e:
          st.warning(f"⚠️ XAI not available: {e}")

# ==============================
# BULK PREDICTION
# ==============================
elif page == "Bulk Prediction":

    st.header("📂 Upload CSV File")

    file = st.file_uploader("Upload CSV", type=["csv"])

    if file is not None:
        try:
            data = pd.read_csv(file)

            st.subheader("📊 Uploaded Data Preview")
            st.dataframe(data.head())

            # Remove target column if exists
            if 'Class' in data.columns:
                data = data.drop('Class', axis=1)

            # Expected format
            expected_columns = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']

            missing_cols = [col for col in expected_columns if col not in data.columns]

            if missing_cols:
                st.error(f"❌ Missing columns: {missing_cols}")
            else:
                data = data[expected_columns]

                # Prediction
                predictions = model.predict(data)
                probabilities = model.predict_proba(data)[:, 1]

                result = data.copy()
                result['Prediction'] = predictions
                result['Fraud_Probability'] = probabilities
                result['Risk_Level'] = [get_risk_level(p) for p in probabilities]

                # ==============================
                # 📊 Dashboard Summary
                # ==============================
                fraud_count = result['Prediction'].sum()
                legit_count = len(result) - fraud_count

                col1, col2 = st.columns(2)
                col1.metric("🚨 Fraud Transactions", fraud_count)
                col2.metric("✅ Legit Transactions", legit_count)

                st.subheader("📋 Results")
                st.dataframe(result.head())

                # ==============================
                # 📈 Chart
                # ==============================
                fig, ax = plt.subplots()

                labels = ['✅ Legit', '🚨 Fraud']
                sizes = [legit_count, fraud_count]

                ax.pie(
                    sizes,
                    labels=labels,
                    autopct='%1.1f%%',
                    startangle=90
                )

                ax.axis('equal')  # Makes it circular

                st.pyplot(fig)

                # ==============================
                # 🔥 XAI - Global Feature Importance
                # ==============================
                st.subheader("🧠 Feature Importance (XAI)")

                try:
                   sample = result.sample(min(100, len(result)))

                   shap_values = explainer(sample[expected_columns])

                   # Handle SHAP output safely
                   if hasattr(shap_values, "values"):
                       values = shap_values.values
                   else:
                       values = shap_values

                   # If classification → take fraud class
                   if len(values.shape) == 3:
                       values = values[:, :, 1]

                   # Mean importance
                   importance = np.abs(values).mean(axis=0)

                   feature_names = expected_columns

                   importance_df = pd.DataFrame({
                       "Feature": feature_names,
                       "Importance": importance
                   }).sort_values(by="Importance", ascending=False)

                   # Plot
                   fig, ax = plt.subplots()
                   ax.barh(importance_df["Feature"], importance_df["Importance"])
                   ax.invert_yaxis()
                   ax.set_title("Feature Importance")

                   st.pyplot(fig)

                except Exception as e:
                    st.warning("⚠️ Unable to generate XAI plot")

                # ==============================
                # 📥 Download Results
                # ==============================
                csv = result.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Download Results CSV",
                    csv,
                    "smartguard_results.csv",
                    "text/csv"
                )

        except Exception as e:
            st.error(f"❌ Error: {e}")

    else:
        st.info("Upload a CSV file to begin.")
