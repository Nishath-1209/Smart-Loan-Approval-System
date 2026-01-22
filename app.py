import streamlit as st
import numpy as np
import joblib

# ================= LOAD MODELS ===================
svm_linear = joblib.load("svm_linear.pkl")
svm_poly   = joblib.load("svm_poly.pkl")
svm_rbf    = joblib.load("svm_rbf.pkl")
scaler = joblib.load("scaler.pkl")

# ================= TITLE ===================
st.set_page_config(page_title="Smart Loan Approval System", layout="centered")

st.title("💰 Smart Loan Approval System")
st.write("This system uses Support Vector Machines to predict loan approval.")

# ================= INPUT SECTION ===================
st.sidebar.header("📋 Enter Applicant Details")

income = st.sidebar.number_input("Applicant Income", 1000, 100000, 5000)
loan = st.sidebar.number_input("Loan Amount", 10, 500, 150)

credit = st.sidebar.selectbox("Credit History", ["Yes", "No"])
employment = st.sidebar.selectbox("Employment Status", ["Salaried", "Self-Employed"])
area = st.sidebar.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# Encode categorical values
credit = 1 if credit == "Yes" else 0
employment = 1 if employment == "Self-Employed" else 0
area_dict = {"Urban": 2, "Semiurban": 1, "Rural": 0}
area = area_dict[area]

# ================= MODEL SELECTION ===================
st.subheader("🧠 Select SVM Kernel Model")

kernel = st.radio("Choose Kernel:", ["Linear SVM", "Polynomial SVM", "RBF SVM"])

if kernel == "Linear SVM":
    model = svm_linear
elif kernel == "Polynomial SVM":
    model = svm_poly
else:
    model = svm_rbf

# ================= PREDICTION ===================
st.subheader("🔍 Loan Eligibility Prediction")

if st.button("Check Loan Eligibility"):

    # Feature Order (12 features exactly like training)
    data = np.array([[1, 1, 1, 0, 1, employment, income, 0, loan, 360, credit, area]])
    
    # Scale input
    data = scaler.transform(data)

    prediction = model.predict(data)[0]
    confidence = model.predict_proba(data)[0][1]

    # ================= OUTPUT ===================
    st.subheader("📊 Prediction Result")

    if prediction == 1:
        st.success("✅ Loan Approved")
        explanation = "Based on credit history and income pattern, the applicant is likely to repay the loan."
        st.write(f"Confidence Score: {confidence:.2f}")
    else:
        st.error("❌ Loan Rejected")
        explanation = "Based on credit history and income pattern, the applicant is unlikely to repay the loan."
        st.write(f"Confidence Score: {1-confidence:.2f}")

    st.info(f"Kernel Used: {kernel}")

    # ================= BUSINESS EXPLANATION ===================
    st.subheader("📘 Business Explanation")
    st.write(explanation)

    st.markdown("""
    **Real Business Meaning:**
    - Banks use ML models to reduce loan default risk.
    - Credit history and income are the most important factors.
    - SVM separates approved and rejected customers using a decision boundary.
    """)

# Footer
st.markdown("---")
st.caption("Developed by S Nishath Tabassum | Smart Loan Approval System Project")
