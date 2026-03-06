import streamlit as st
import pickle
import pandas as pd

from fraud_detection_model import detect_fraud

# Load model
model = pickle.load(open("credit_risk_model.pkl", "rb"))

st.title("AI Credit Risk + Fraud Detection")

st.header("Credit Risk Prediction")

income = st.number_input("Income", 1000, 200000)
employment_length = st.number_input("Employment Length (years)", 0, 40)
debt_ratio = st.slider("Debt Ratio", 0.0, 1.0)
credit_score = st.slider("Credit Score", 300, 850)

loan_purpose = st.selectbox(
    "Loan Purpose",
    ["car", "house", "education", "business"]
)

# Encode loan purpose
purpose_map = {
    "car": 0,
    "house": 1,
    "education": 2,
    "business": 3
}

purpose_encoded = purpose_map[loan_purpose]

if st.button("Predict Credit Risk"):

    input_data = pd.DataFrame([[
        income,
        employment_length,
        debt_ratio,
        credit_score,
        purpose_encoded
    ]],
    columns=[
        "income",
        "employment_length",
        "debt_ratio",
        "credit_score",
        "loan_purpose"
    ])

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.error("⚠️ High Risk - Loan not recommended")
    else:
        st.success("✅ Low Risk - Loan approved")

# Fraud detection section
st.header("Fraud Detection")

amount = st.number_input("Transaction Amount", 1, 10000)
frequency = st.slider("Transaction Frequency", 0.0, 1.0)
location = st.slider("Location Change Score", 0.0, 1.0)

if st.button("Check Fraud"):

    result = detect_fraud(amount, frequency, location)

    st.write(result)
