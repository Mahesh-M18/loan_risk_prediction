import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Loan Risk Prediction", page_icon="🏦", layout="wide")

# Load model & preprocessor
@st.cache_resource(show_spinner=False)
def load_artifacts():
    model_path = "results/rf.joblib"
    preproc_path = "results/preprocessor.joblib"
    if not (os.path.exists(model_path) and os.path.exists(preproc_path)):
        st.warning("Model not found. Please run training first: python src/train_from_synthetic.py")
        return None, None
    model_obj = joblib.load(model_path)
    label_enc = joblib.load(preproc_path)
    return model_obj, label_enc

model, label_encoder = load_artifacts()

st.markdown("## Loan Risk Prediction App")
st.caption("Provide applicant details to estimate the probability of loan default.")

# Sidebar with model info
# with st.sidebar:
#     st.header("Model")
#     if model is not None:
#         if hasattr(model, "n_estimators"):
#             st.write(f"Estimators: {getattr(model, 'n_estimators', 'N/A')}")
#         if hasattr(model, "feature_names_in_"):
#             st.write("Features:")
#             for name in model.feature_names_in_:
#                 st.caption(f"• {name}")
#     else:
#         st.warning("No model loaded.")

with st.container():
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=30, help="Applicant age in years")
        years_employed = st.number_input("Years Employed", min_value=0, max_value=60, value=5)
    with col2:
        annual_income = st.number_input("Annual Income", min_value=1000, max_value=2_000_000, value=50_000, step=500)
        dependents = st.number_input("Dependents", min_value=0, max_value=10, value=0)
    with col3:
        loan_amount = st.number_input("Loan Amount", min_value=500, max_value=1_000_000, value=15_000, step=500)
        loan_tenure_months = st.number_input("Loan Tenure (months)", min_value=12, max_value=120, value=36, step=1)

credit_history = st.radio("Credit History", ["Yes", "No"], horizontal=True)

predict_clicked = st.button("Predict Risk", type="primary", disabled=(model is None))

def build_input_dataframe() -> pd.DataFrame:
    encoded_history = label_encoder.transform([credit_history])[0]
    row = {
        "Age": age,
        "Annual_Income": annual_income,
        "Loan_Amount": loan_amount,
        "Years_Employed": years_employed,
        "Dependents": dependents,
        "Credit_History": encoded_history,
        "Loan_Tenure_Months": loan_tenure_months,
    }
    df = pd.DataFrame([row])
    # Reorder to match training
    if hasattr(model, "feature_names_in_"):
        cols = list(model.feature_names_in_)
        df = df.reindex(columns=cols)
    return df

result_col1, result_col2 = st.columns([2, 1], gap="large")

if predict_clicked and model is not None:
    input_df = build_input_dataframe()
    proba = model.predict_proba(input_df)[0]
    prediction = int(model.predict(input_df)[0])
    risk_percentage = round(float(proba[1]) * 100.0, 2)
    risk_label = "High Risk" if prediction == 1 else "Low Risk"

    with result_col1:
        # Affordability metrics and rule-based override
        monthly_income = annual_income / 12.0
        monthly_installment = loan_amount / max(loan_tenure_months, 1)
        payment_ratio = monthly_installment / max(monthly_income, 1)
        ratio_threshold = 0.75
        rule_high = payment_ratio > ratio_threshold
        final_prediction = 1 if rule_high else prediction
        final_label = "High Risk" if final_prediction == 1 else "Low Risk"

        st.markdown(f"### Prediction: {final_label}")
        st.progress(int(risk_percentage))
        st.write(f"Estimated probability of default: **{risk_percentage}%**")
        if final_prediction == 1:
            if rule_high and prediction == 0:
                st.error("High risk due to Payment/Income > 0.75×.")
            else:
                st.error("This applicant appears to be at HIGH risk of loan default.")
        else:
            st.success("This applicant appears to be at LOW risk of loan default.")

        with st.expander("View input summary"):
            show_df = input_df.copy()
            show_df["Credit_History"] = credit_history
            st.dataframe(show_df, use_container_width=True)

        # Quick affordability metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Monthly installment", f"${monthly_installment:,.0f}")
        m2.metric("Monthly income", f"${monthly_income:,.0f}")
        m3.metric("Payment / Income", f"{payment_ratio:.2f}x")

    # with result_col2:
    #     if os.path.exists("results/confusion_matrix.png"):
    #         st.caption("Model confusion matrix")
    #         st.image("results/confusion_matrix.png", use_column_width=True)
    #     else:
    #         st.caption("Train the model to view confusion matrix.")

    #     if hasattr(model, "feature_importances_") and hasattr(model, "feature_names_in_"):
    #         st.caption("Top feature importances")
    #         importances = pd.Series(model.feature_importances_, index=model.feature_names_in_)
    #         top_importances = importances.sort_values(ascending=True).tail(7)
    #         st.bar_chart(top_importances)

st.markdown("---")
with st.expander("About this app"):
    st.write(
        "This app uses a RandomForest model trained on synthetic or provided data. "
        "Features include Age, Annual Income, Loan Amount, Years Employed, Dependents, Credit History, and Loan Tenure (months)."
    )
