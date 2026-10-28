import streamlit as st
import pandas as pd
from prediction_helper import predict

# Page Configuration - Clean, Standard Title and Layout
st.set_page_config(
    page_title="Credit Risk Modeling & Scoring Engine",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Light Professional Theme (Clean Whites, Soft Slate, Subtle Borders)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    /* Enforce Pure Light Theme */
    html, body, [class*="css"], .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        background-color: #ffffff !important;
        color: #1e293b !important;
    }

    /* Clean Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #f8fafc !important;
        border-right: 1px solid #e2e8f0 !important;
    }
    section[data-testid="stSidebar"] h3 {
        color: #0f172a !important;
        font-weight: 600 !important;
    }

    /* Header Box */
    .header-box {
        padding: 1.25rem 0 1.25rem 0;
        border-bottom: 1px solid #e2e8f0;
        margin-bottom: 1.75rem;
    }
    .header-title {
        font-size: 1.65rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 0.25rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        letter-spacing: -0.3px;
    }
    .header-subtitle {
        font-size: 0.95rem;
        color: #64748b;
        margin: 0;
        font-weight: 400;
    }

    /* Section Subheading */
    .section-title {
        font-size: 0.85rem;
        font-weight: 700;
        color: #475569;
        margin-bottom: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }

    /* Clean Grouping Container */
    .form-group-card {
        background-color: #fafafa;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 1.25rem;
        margin-bottom: 1.25rem;
    }

    /* Professional Button */
    div.stButton > button:first-child {
        background-color: #2563eb !important;
        color: #ffffff !important;
        border: 1px solid #2563eb !important;
        border-radius: 8px !important;
        padding: 0.65rem 1.5rem !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05) !important;
        transition: all 0.15s ease-in-out !important;
    }
    div.stButton > button:first-child:hover {
        background-color: #1d4ed8 !important;
        border-color: #1d4ed8 !important;
        color: #ffffff !important;
    }

    /* Result Dashboard Container */
    .result-container {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.75rem;
        margin-top: 1.5rem;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.03);
    }

    .score-card {
        text-align: center;
        padding: 1.25rem 1rem;
        border-radius: 10px;
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
    }
    .score-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        color: #64748b;
        font-weight: 600;
        margin-bottom: 0.35rem;
    }
    .score-number {
        font-size: 2.25rem;
        font-weight: 700;
        color: #0f172a;
        line-height: 1.1;
    }
    .score-scale {
        font-size: 0.8rem;
        color: #94a3b8;
        margin-top: 0.25rem;
    }

    /* Soft Professional Badges */
    .badge-pill {
        display: inline-block;
        padding: 0.35rem 0.9rem;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-excellent {
        background-color: #ecfdf5;
        color: #047857;
        border: 1px solid #a7f3d0;
    }
    .badge-good {
        background-color: #f0fdf4;
        color: #15803d;
        border: 1px solid #bbf7d0;
    }
    .badge-average {
        background-color: #fffbeb;
        color: #b45309;
        border: 1px solid #fde68a;
    }
    .badge-poor {
        background-color: #fef2f2;
        color: #b91c1c;
        border: 1px solid #fecaca;
    }

    /* Soft Decision Directives */
    .decision-banner {
        padding: 1rem 1.25rem;
        border-radius: 8px;
        margin-top: 1.25rem;
        font-size: 0.95rem;
        font-weight: 500;
    }
    .decision-excellent {
        background-color: #f0fdf4;
        color: #166534;
        border-left: 4px solid #22c55e;
    }
    .decision-good {
        background-color: #f0fdf4;
        color: #166534;
        border-left: 4px solid #16a34a;
    }
    .decision-average {
        background-color: #fffbeb;
        color: #92400e;
        border-left: 4px solid #f59e0b;
    }
    .decision-poor {
        background-color: #fef2f2;
        color: #991b1b;
        border-left: 4px solid #ef4444;
    }

    /* Clean Subtle Metric Chip */
    .metric-inline {
        padding: 0.65rem 0.85rem;
        border-radius: 8px;
        background-color: #f1f5f9;
        font-size: 0.85rem;
        color: #475569;
        border: 1px solid #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.markdown("### 💳 Quick Presets")
    st.caption("Select a sample profile to quickly populate the form.")
    
    preset_choice = st.selectbox(
        "Borrower Profile:",
        [
            "Custom Application",
            "Low Risk Profile",
            "Moderate Risk Profile",
            "Elevated Risk Profile"
        ],
        index=0
    )

    st.markdown("---")
    st.markdown("### 📋 Application Guidelines")
    st.markdown("""
    - **Score Range**: 300 to 900.
    - **Healthy Utilization**: Below 30% credit card usage is ideal.
    - **Loan to Income**: A lower ratio signifies better repayment capacity.
    """)
    st.markdown("---")
    st.caption("Credit Risk Modeling & Scoring Engine")

# Map presets cleanly
if preset_choice == "Low Risk Profile":
    def_age, def_income, def_loan = 38, 2400000, 1200000
    def_tenure, def_dpd, def_delinq = 24, 0, 0
    def_util, def_open_acc = 15, 2
    def_res, def_purp, def_type = "Owned", "Home", "Secured"
elif preset_choice == "Moderate Risk Profile":
    def_age, def_income, def_loan = 29, 1200000, 2000000
    def_tenure, def_dpd, def_delinq = 36, 10, 15
    def_util, def_open_acc = 38, 3
    def_res, def_purp, def_type = "Rented", "Auto", "Secured"
elif preset_choice == "Elevated Risk Profile":
    def_age, def_income, def_loan = 24, 500000, 1800000
    def_tenure, def_dpd, def_delinq = 48, 40, 50
    def_util, def_open_acc = 75, 4
    def_res, def_purp, def_type = "Rented", "Personal", "Unsecured"
else:
    def_age, def_income, def_loan = 28, 1200000, 2500000
    def_tenure, def_dpd, def_delinq = 36, 15, 20
    def_util, def_open_acc = 30, 2
    def_res, def_purp, def_type = "Owned", "Education", "Unsecured"

# ----------------- MAIN HEADER -----------------
st.markdown("""
<div class="header-box">
    <div class="header-title">💳 Credit Risk Modeling & Scoring Engine</div>
    <p class="header-subtitle">Evaluate loan applicant creditworthiness, calculate credit scores, and review loan eligibility.</p>
</div>
""", unsafe_allow_html=True)

# ----------------- CLEAN INPUT FORM -----------------
col_applicant, col_bureau = st.columns([1, 1], gap="large")

with col_applicant:
    st.markdown("""
    <div class="section-title">👤 Applicant & Loan Details</div>
    """, unsafe_allow_html=True)
    
    r1_c1, r1_c2 = st.columns(2)
    with r1_c1:
        age = st.number_input("Age", min_value=18, max_value=100, value=def_age, step=1)
        residence_type = st.selectbox("Residence Type", ["Owned", "Rented", "Mortgage"], index=["Owned", "Rented", "Mortgage"].index(def_res))
    with r1_c2:
        income = st.number_input("Annual Income (₹)", min_value=10000, max_value=50000000, value=def_income, step=50000)
        loan_purpose = st.selectbox("Loan Purpose", ["Home", "Auto", "Education", "Personal"], index=["Home", "Auto", "Education", "Personal"].index(def_purp))

    r2_c1, r2_c2 = st.columns(2)
    with r2_c1:
        loan_amount = st.number_input("Loan Amount (₹)", min_value=10000, max_value=100000000, value=def_loan, step=50000)
        loan_type = st.selectbox("Loan Type", ["Secured", "Unsecured"], index=["Secured", "Unsecured"].index(def_type))
    with r2_c2:
        loan_tenure_months = st.number_input("Tenure (Months)", min_value=6, max_value=360, value=def_tenure, step=6)
        
        # Clean Loan to Income Ratio
        lti_ratio = loan_amount / income if income > 0 else 0
        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="metric-inline">
            Loan to Income: <strong>{lti_ratio:.2f}x</strong>
        </div>
        """, unsafe_allow_html=True)

with col_bureau:
    st.markdown("""
    <div class="section-title">📊 Credit & Repayment History</div>
    """, unsafe_allow_html=True)

    b1_c1, b1_c2 = st.columns(2)
    with b1_c1:
        credit_utilization_ratio = st.slider("Credit Card Utilization (%)", min_value=0, max_value=100, value=def_util, step=1)
        delinquency_ratio = st.slider("Past Delinquency Rate (%)", min_value=0, max_value=100, value=def_delinq, step=1)
    with b1_c2:
        avg_dpd_per_delinquency = st.number_input("Avg Overdue Days (DPD)", min_value=0, max_value=365, value=def_dpd, step=1)
        num_open_accounts = st.number_input("Active Credit Accounts", min_value=1, max_value=10, value=def_open_acc, step=1)

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
    st.caption("Tip: Applicants with consistent on-time payments and utilization below 30% typically achieve higher credit scores.")

# ----------------- ACTION BUTTON -----------------
st.markdown("<br/>", unsafe_allow_html=True)
evaluate_click = st.button("Calculate Credit Score", type="primary", use_container_width=True)

# ----------------- RESULT SECTION -----------------
if evaluate_click:
    with st.spinner("Calculating credit score..."):
        probability, credit_score, rating = predict(
            age, income, loan_amount, loan_tenure_months, avg_dpd_per_delinquency,
            delinquency_ratio, credit_utilization_ratio, num_open_accounts,
            residence_type, loan_purpose, loan_type
        )

    # Clean, Standard Tier Styling (Soft Tones)
    if rating == "Excellent":
        badge_html = '<span class="badge-pill badge-excellent">Excellent Credit</span>'
        decision_class = "decision-excellent"
        decision_text = "Approved — Prime eligibility with preferred terms and standard processing."
    elif rating == "Good":
        badge_html = '<span class="badge-pill badge-good">Good Credit</span>'
        decision_class = "decision-good"
        decision_text = "Approved — Qualifies for standard loan products and terms."
    elif rating == "Average":
        badge_html = '<span class="badge-pill badge-average">Moderate Credit</span>'
        decision_class = "decision-average"
        decision_text = "Review Recommended — Standard approval subject to additional document or income verification."
    else:
        badge_html = '<span class="badge-pill badge-poor">High Risk</span>'
        decision_class = "decision-poor"
        decision_text = "Eligibility Criteria Not Met — High likelihood of default based on current profile."

    st.markdown('<div class="result-container">', unsafe_allow_html=True)
    
    # Clean Top Metrics Row
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(f"""
        <div class="score-card">
            <div class="score-label">Credit Score</div>
            <div class="score-number">{credit_score}</div>
            <div class="score-scale">Scale: 300 to 900</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="score-card">
            <div class="score-label">Risk Category</div>
            <div style="margin-top: 0.5rem; margin-bottom: 0.5rem;">{badge_html}</div>
            <div class="score-scale">Tier Assessment</div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="score-card">
            <div class="score-label">Estimated Risk</div>
            <div class="score-number">{probability:.1%}</div>
            <div class="score-scale">Default Likelihood</div>
        </div>
        """, unsafe_allow_html=True)

    # Score Position Bar
    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    progress_val = max(0.0, min(1.0, (credit_score - 300) / 600))
    st.progress(progress_val)
    
    col_p1, col_p2, col_p3, col_p4 = st.columns(4)
    col_p1.caption("300 — Poor")
    col_p2.caption("500 — Average")
    col_p3.caption("650 — Good")
    col_p4.caption("750 to 900 — Excellent")

    # Simple Human Decision Banner
    st.markdown(f"""
    <div class="decision-banner {decision_class}">
        <strong>Loan Assessment:</strong> {decision_text}
    </div>
    """, unsafe_allow_html=True)

    # Clean Observations Summary (Human Friendly)
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    with st.expander("Summary Observations", expanded=True):
        obs_col1, obs_col2 = st.columns(2)
        with obs_col1:
            st.markdown("**Favorable Highlights:**")
            positives = []
            if lti_ratio < 2.5:
                positives.append(f"Manageable loan-to-income balance ({lti_ratio:.2f}x)")
            if credit_utilization_ratio <= 30:
                positives.append(f"Healthy credit utilization ({credit_utilization_ratio}%)")
            if delinquency_ratio == 0:
                positives.append("Clean past repayment record")
            if residence_type == "Owned":
                positives.append("Homeowner stability")
            if loan_type == "Secured":
                positives.append("Secured loan structure")

            if positives:
                for p in positives:
                    st.markdown(f"- {p}")
            else:
                st.markdown("- Standard application parameters")

        with obs_col2:
            st.markdown("**Points to Note:**")
            notes = []
            if lti_ratio >= 3.5:
                notes.append(f"High loan request relative to annual income ({lti_ratio:.2f}x)")
            if credit_utilization_ratio > 50:
                notes.append(f"High revolving credit card balance ({credit_utilization_ratio}%)")
            if delinquency_ratio > 15:
                notes.append(f"Past overdue repayment history ({delinquency_ratio}%)")
            if avg_dpd_per_delinquency >= 30:
                notes.append(f"Significant overdue delay ({avg_dpd_per_delinquency} days past due)")
            if loan_type == "Unsecured":
                notes.append("Unsecured lending category")

            if notes:
                for n in notes:
                    st.markdown(f"- {n}")
            else:
                st.markdown("- No significant risk concerns noted")

    st.markdown('</div>', unsafe_allow_html=True)
