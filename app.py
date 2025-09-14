import streamlit as st
import pandas as pd
import joblib

# Load the saved Logistic Regression model
model = joblib.load("loan_model.pkl")



st.title("🏦 Loan Approval Prediction App")
st.write("Fill in the details to check if your loan will be approved or not.")

# Input fields
gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
applicant_income = st.number_input("Applicant Income", min_value=0)
coapplicant_income=st.number_input("Coapplicant Income",min_value=0)
loan_amount = st.number_input("Loan Amount", min_value=0)
loan_amount_term = st.number_input("Loan Amount Term (in months)", min_value=12)
credit_history = st.selectbox("Credit History", ["Yes","No"])
property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

def check_loan_conditions(loan_amount, loan_amount_term, applicant_income, coapplicant_income):
    total_income = applicant_income + coapplicant_income
    monthly_income = total_income 
    emi = loan_amount / loan_amount_term
    monthly_limit = 0.75 * monthly_income

    if loan_amount < total_income:
        return "✅ Loan Auto-Approved: Loan amount less than total income."
    if emi >= monthly_limit:
        return "❌ Loan Auto-Rejected: EMI exceeds 75% of monthly income."

    # None means no auto decision; continue to model prediction
    return None

# Encode inputs (must match training preprocessing)
data = {
    "Gender": 1 if gender == "Male" else 0,
    "Married": 1 if married == "Yes" else 0,
    "Dependents": 3 if dependents == "3+" else int(dependents),
    "Education": 0 if education == "Graduate" else 1,
    "Self_Employed": 1 if self_employed == "Yes" else 0,
    "TotalIncome": applicant_income+coapplicant_income,
    "LoanAmount": loan_amount // 1000,
    "Loan_Amount_Term": loan_amount_term,
    "Credit_History": 1 if credit_history=="Yes" else 0,
    "Property_Area": 0 if property_area == "Rural" else (1 if property_area == "Semiurban" else 2),
}

input_df = pd.DataFrame([data])

# Predict button
if st.button("Predict Loan Status"):
    precheck_msg = check_loan_conditions(loan_amount, loan_amount_term, applicant_income, coapplicant_income)
    if precheck_msg:
        st.info(precheck_msg)
    else:
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]  # probability of approval

        if prediction == 1:
            st.success(f"✅ Loan Approved (Confidence: {probability*100:.2f}%)")
        else:
            st.error(f"❌ Loan Not Approved (Confidence: {(1-probability)*100:.2f}%)")
