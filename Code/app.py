import streamlit as st
import pandas as pd
import numpy as np
from scan_customers import LoanScreener

# PEPE Style: Load the model once to save memory and improve performance
@st.cache_resource
def load_screener():
    # Initialize the screener using your specific saved XGBoost models
    return LoanScreener(models_path='Models')

screener = load_screener()

st.set_page_config(page_title="Internal Risk Portal", layout="wide")
st.title("🏦 Internal Risk Assessment Portal")
st.markdown("---")

# --- STAGE 1: GATEKEEPER (MANDATORY) ---
st.subheader("Stage 1: Eligibility (Gatekeeper Features)")
col1, col2, col3 = st.columns(3)

with col1:
    loan_amount = st.number_input("Loan Amount ($)", min_value=1000, value=15000, step=500)
    annual_inc = st.number_input("Annual Income ($)", min_value=5000, value=75000, step=1000)

with col2:
    # Ensuring case consistency for XGBoost categorical Dtypes
    addr_state = st.text_input("State (e.g., FL, CA, NY)", value="FL").upper()
    zip_code = st.text_input("Zip Code (3-digit prefix)", value="331")

with col3:
    # Use standard LendingClub purpose categories
    title_options = ["Debt consolidation", "Credit card", "Home improvement", "Major purchase", "Other"]
    title = st.selectbox("Loan Purpose", title_options)

# --- STAGE 2: DETAILED PROFILE (OPTIONAL) ---
st.subheader("Stage 2: Risk Profile (41 Features)")
with st.expander("Expand to enter detailed credit data for PD Estimation"):
    st.info("Empty fields will use historical medians or derived estimates to prevent model bias.")
    
    c1, c2, c3 = st.columns(3)
    with c1:
        fico = st.number_input("FICO Score", 300, 850, value=None, help="If blank, defaults to 700")
        emp_length = st.number_input("Employment Length (Years)", 0, 45, value=None)
        revol_util = st.number_input("Revol. Utilization (%)", 0.0, 100.0, value=None)
    
    with c2:
        revol_bal = st.number_input("Revolving Balance ($)", min_value=0, value=None)
        total_acc = st.number_input("Total Credit Accounts", min_value=0, value=None)
        open_acc = st.number_input("Open Credit Lines", min_value=0, value=None)
        
    with c3:
        mort_acc = st.number_input("Mortgage Accounts", min_value=0, value=None)
        pub_rec = st.number_input("Public Records / Bankruptcies", min_value=0, value=None)
        inq_last_6m = st.number_input("Inquiries (Last 6m)", 0, 10, value=None)

# --- EXECUTION LOGIC ---
if st.button("Execute Credit Decision", use_container_width=True):
    
    # 1. Automated DTI calculation
    # Mathematical Finance: DTI is a primary risk driver; we estimate monthly debt
    # based on the requested loan and a baseline household expense.
    monthly_inc = annual_inc / 12 if annual_inc > 0 else 1
    est_installment = loan_amount / 36  # Simplified 3-year term estimate
    # 500 represents a baseline monthly debt floor for non-blank applicants
    dti = ((est_installment + 500) / monthly_inc) * 100

    # 2. Build the full 41+ feature DataFrame
    all_features = list(set(screener.gatekeeper_features + screener.risk_features))
    df_applicant = pd.DataFrame(0, index=[0], columns=all_features)

    # 3. Inject Stage 1 Inputs (Force Dtypes for XGBoost Categories)
    df_applicant['loan_amnt'] = loan_amount
    df_applicant['annual_inc'] = annual_inc
    df_applicant['dti'] = dti
    df_applicant['addr_state'] = addr_state
    df_applicant['zip_code'] = zip_code
    df_applicant['title'] = title

    # 4. Inject Stage 2 Inputs with Contextual Defaults
    # If the employee leaves these blank, we use 'Safe' medians to stabilize the model.
    df_applicant['fico_range_low'] = fico if fico is not None else 700
    df_applicant['fico_range_high'] = (fico + 4) if fico is not None else 704
    df_applicant['emp_length'] = emp_length if emp_length is not None else 5
    df_applicant['revol_util'] = revol_util if revol_util is not None else 35.0
    df_applicant['total_acc'] = total_acc if total_acc is not None else 15
    df_applicant['revol_bal'] = revol_bal if revol_bal is not None else (annual_inc * 0.15)
    
    # Fill remaining features with 0 (or 'Unknown' for categories)
    for col in all_features:
        if pd.isna(df_applicant[col][0]):
            df_applicant[col] = 0

    # 5. Pipeline Execution
    with st.spinner("Analyzing applicant risk profile..."):
        decision = screener.screen_customer(df_applicant)
    
    # 6. Final UI Output
    st.markdown("---")
    if decision["Status"] == "APPROVED":
        st.success(f"**DECISION: {decision['Status']}**")
        r1, r2, r3 = st.columns(3)
        r1.metric("Internal Grade", decision["Assigned Grade"])
        r2.metric("Interest Rate", decision["Interest Rate"])
        r3.metric("Probability of Default", decision["Probability of Default"])
    else:
        st.error(f"**DECISION: {decision['Status']}**")
        st.warning(f"Reason: {decision.get('Reason', 'High Risk Profile Detected')}")
        st.write(f"Calculated Debt-to-Income (DTI): {dti:.2f}%")

st.markdown("---")
st.caption("Internal Tool - Authorized Personnel Only. Risk Models trained on LendingClub Historical Data.")