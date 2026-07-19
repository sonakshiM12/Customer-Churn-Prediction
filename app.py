import streamlit as st
import pandas as pd
import joblib

# ============================
# Load Saved Files
# ============================

model = joblib.load("model/churn_model.pkl")
encoders = joblib.load("model/encoders.pkl")
features = joblib.load("model/features.pkl")

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)

st.title("📊 Customer Churn Prediction System")

st.write(
    "Enter customer details below and click **Predict**."
)

st.markdown("---")

# ============================
# User Inputs
# ============================

gender = st.selectbox(
    "Gender",
    list(encoders["gender"].classes_)
)

senior = st.selectbox(
    "Senior Citizen",
    [0, 1]
)

partner = st.selectbox(
    "Partner",
    list(encoders["Partner"].classes_)
)

dependents = st.selectbox(
    "Dependents",
    list(encoders["Dependents"].classes_)
)

tenure = st.slider(
    "Tenure (Months)",
    0,
    72,
    12
)

phone = st.selectbox(
    "Phone Service",
    list(encoders["PhoneService"].classes_)
)

multiple = st.selectbox(
    "Multiple Lines",
    list(encoders["MultipleLines"].classes_)
)

internet = st.selectbox(
    "Internet Service",
    list(encoders["InternetService"].classes_)
)

online_security = st.selectbox(
    "Online Security",
    list(encoders["OnlineSecurity"].classes_)
)

online_backup = st.selectbox(
    "Online Backup",
    list(encoders["OnlineBackup"].classes_)
)

device = st.selectbox(
    "Device Protection",
    list(encoders["DeviceProtection"].classes_)
)

tech = st.selectbox(
    "Tech Support",
    list(encoders["TechSupport"].classes_)
)

stream_tv = st.selectbox(
    "Streaming TV",
    list(encoders["StreamingTV"].classes_)
)

stream_movies = st.selectbox(
    "Streaming Movies",
    list(encoders["StreamingMovies"].classes_)
)

contract = st.selectbox(
    "Contract",
    list(encoders["Contract"].classes_)
)

paperless = st.selectbox(
    "Paperless Billing",
    list(encoders["PaperlessBilling"].classes_)
)

payment = st.selectbox(
    "Payment Method",
    list(encoders["PaymentMethod"].classes_)
)

monthly = st.number_input(
    "Monthly Charges",
    min_value=0.0,
    value=70.0
)

total = st.number_input(
    "Total Charges",
    min_value=0.0,
    value=1000.0
)
# ============================
# Prediction
# ============================

if st.button("Predict Churn"):

    # Encode categorical values
    data = {
        "gender": encoders["gender"].transform([gender])[0],
        "SeniorCitizen": senior,
        "Partner": encoders["Partner"].transform([partner])[0],
        "Dependents": encoders["Dependents"].transform([dependents])[0],
        "tenure": tenure,
        "PhoneService": encoders["PhoneService"].transform([phone])[0],
        "MultipleLines": encoders["MultipleLines"].transform([multiple])[0],
        "InternetService": encoders["InternetService"].transform([internet])[0],
        "OnlineSecurity": encoders["OnlineSecurity"].transform([online_security])[0],
        "OnlineBackup": encoders["OnlineBackup"].transform([online_backup])[0],
        "DeviceProtection": encoders["DeviceProtection"].transform([device])[0],
        "TechSupport": encoders["TechSupport"].transform([tech])[0],
        "StreamingTV": encoders["StreamingTV"].transform([stream_tv])[0],
        "StreamingMovies": encoders["StreamingMovies"].transform([stream_movies])[0],
        "Contract": encoders["Contract"].transform([contract])[0],
        "PaperlessBilling": encoders["PaperlessBilling"].transform([paperless])[0],
        "PaymentMethod": encoders["PaymentMethod"].transform([payment])[0],
        "MonthlyCharges": monthly,
        "TotalCharges": total
    }

    # Create dataframe
    input_df = pd.DataFrame([data])

    # Arrange columns in same order used during training
    input_df = input_df[features]

    # Prediction
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]

    st.markdown("---")
    st.subheader("Prediction Result")

    if prediction == encoders["Churn"].transform(["Yes"])[0]:
        st.error("⚠️ Customer is likely to Churn")
    else:
        st.success("✅ Customer is NOT likely to Churn")

    st.write(f"Confidence: **{max(probability) * 100:.2f}%**")

    st.subheader("Prediction Probabilities")

    classes = encoders["Churn"].inverse_transform(model.classes_)

    prob_df = pd.DataFrame({
        "Class": classes,
        "Probability": probability
    })

    st.dataframe(prob_df, use_container_width=True)

st.markdown("---")
st.caption("Customer Churn Prediction using Random Forest | Built with Streamlit")