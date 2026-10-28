import os
import streamlit as st
import pandas as pd
from prediction_helper import predict

# Page configuration
st.set_page_config(
    page_title="Credit Risk Modeling & Scoring Engine",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End CSS Styling
st.markdown("""
<style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Header Card */
    .hero-container {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f766e 100%);
        padding: 2.2rem 2.5rem;
        border-radius: 18px;
        color: #ffffff;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.35);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin-bottom: 0.5rem;
        color: #f8fafc;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #cbd5e1;
        font-weight: 400;
        line-height: 1.5;
        margin-bottom: 1.2rem;
    }
    .badge-pill {
        display: inline-block;
        padding: 0.3rem 0.85rem;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-right: 0.5rem;
        margin-bottom: 0.3rem;
        background: rgba(255, 255, 255, 0.12);
        color: #e2e8f0;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    .badge-pill-accent {
        background: rgba(45, 212, 191, 0.18);
        color: #5eead4;
        border: 1px solid rgba(45, 212, 191, 0.35);
    }

    /* Section Card Containers */
    .section-card {
        background: #ffffff;
        padding: 1.6rem 1.8rem;
        border-radius: 14px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.03);
        margin-bottom: 1.5rem;
    }
    @media (prefers-color-scheme: dark) {
        .section-card {
            background: #1e293b;
            border-color: #334155;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
        }
    }
    .section-header {
        font-size: 1.15rem;
        font-weight: 700;
        margin-bottom: 1.2rem;
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }

    /* Result Dashboard Cards */
    .metric-card {
        padding: 1.5rem;
        border-radius: 14px;
        text-align: center;
        margin-bottom: 1rem;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.04);
        background: #ffffff;
    }
    @media (prefers-color-scheme: dark) {
        .metric-card {
            background: #1e293b;
            border-color: #334155;
        }
    }
    .metric-title {
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        color: #64748b;
        margin-bottom: 0.5rem;
    }
    .metric-value {
        font-size: 2.3rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin-bottom: 0.3rem;
    }

    /* Rating specific highlights */
    .rating-excellent {
        color: #10b981;
        background: rgba(16, 185, 129, 0.08);
        border: 1.5px solid #10b981;
    }
    .rating-good {
        color: #3b82f6;
        background: rgba(59, 130, 246, 0.08);
        border: 1.5px solid #3b82f6;
    }
    .rating-average {
        color: #f59e0b;
        background: rgba(245, 158, 11, 0.08);
        border: 1.5px solid #f59e0b;
    }
    .rating-poor {
        color: #ef4444;
        background: rgba(239, 68, 68, 0.08);
        border: 1.5px solid #ef4444;
    }

    /* Live ratio badge */
    .ratio-box {
        padding: 0.75rem 1rem;
        border-radius: 10px;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        margin-top: 0.5rem;
    }
    @media (prefers-color-scheme: dark) {
        .ratio-box {
            background: #0f172a;
            border-color: #334155;
        }
    }
</style>
""", unsafe_allow_html=True)

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.markdown("### 🏦 Credit Risk Engine")
    st.caption("AI-Powered Risk Decisioning v1.2")
    st.markdown("---")

    st.subheader("⚡ Quick Applicant Presets")
    preset_choice = st.selectbox(
        "Load Sample Applicant Profile:",
        [
            "Custom Manual Entry",
            "🌟 Prime Borrower (Low Risk)",
            "⚖️ Near-Prime Applicant (Moderate Risk)",
            "⚠️ Subprime Applicant (High Risk)"
        ],
        index=0
    )

    st.markdown("---")
    st.subheader("📌 System Specifications")
    st.markdown("""
    - **Engine**: Logistic Regression Scorecard
    - **Optimization**: Optuna Bayesian Search
    - **Sampling**: SMOTE-Tomek Links
    - **ROC-AUC**: `0.984` | **Gini**: `0.967`
    - **Target Metric**: Probability of Default (PD)
    - **Score Range**: 300 to 900
    """)
    st.markdown("---")
    st.caption("© Credit Risk Modeling & Scoring Engine. All rights reserved.")

# Preset Default Value Mappings
if preset_choice == "🌟 Prime Borrower (Low Risk)":
    def_age, def_income, def_loan = 38, 2800000, 1500000
    def_tenure, def_dpd, def_delinq = 24, 0, 0
    def_util, def_open_acc = 12, 2
    def_res, def_purp, def_type = "Owned", "Home", "Secured"
elif preset_choice == "⚖️ Near-Prime Applicant (Moderate Risk)":
    def_age, def_income, def_loan = 29, 1200000, 2400000
    def_tenure, def_dpd, def_delinq = 36, 15, 12
    def_util, def_open_acc = 42, 3
    def_res, def_purp, def_type = "Rented", "Auto", "Secured"
elif preset_choice == "⚠️ Subprime Applicant (High Risk)":
    def_age, def_income, def_loan = 23, 480000, 2200000
    def_tenure, def_dpd, def_delinq = 48, 55, 65
    def_util, def_open_acc = 82, 4
    def_res, def_purp, def_type = "Rented", "Personal", "Unsecured"
else:
    def_age, def_income, def_loan = 28, 1200000, 2560000
    def_tenure, def_dpd, def_delinq = 36, 20, 30
    def_util, def_open_acc = 30, 2
    def_res, def_purp, def_type = "Owned", "Education", "Unsecured"

# ----------------- HERO HEADER -----------------
st.markdown("""
<div class="hero-container">
    <div class="hero-title">🏦 Credit Risk Modeling & Scoring Engine</div>
    <div class="hero-subtitle">
        Enterprise-grade risk assessment platform translating borrower behavioral history, leverage, and demographics into calibrated Credit Scores and automated underwriting recommendations.
    </div>
    <div>
        <span class="badge-pill badge-pill-accent">● Production Model v1.2</span>
        <span class="badge-pill">ROC-AUC: 0.984</span>
        <span class="badge-pill">Gini: 0.967</span>
        <span class="badge-pill">SMOTE-Tomek Calibrated</span>
        <span class="badge-pill">Basel II Compliant</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ----------------- INPUT FORM -----------------
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown("### 👤 1. Applicant Profile & Loan Terms")
    
    with st.container():
        subcol1, subcol2 = st.columns(2)
        with subcol1:
            age = st.number_input(
                "Applicant Age",
                min_value=18,
                max_value=100,
                value=def_age,
                step=1,
                help="Applicant age in completed years."
            )
            residence_type = st.selectbox(
                "Residence Status",
                ["Owned", "Rented", "Mortgage"],
                index=["Owned", "Rented", "Mortgage"].index(def_res),
                help="Current residential stability indicator."
            )
        with subcol2:
            income = st.number_input(
                "Annual Income (₹)",
                min_value=10000,
                max_value=50000000,
                value=def_income,
                step=50000,
                help="Total verified gross annual income."
            )
            loan_purpose = st.selectbox(
                "Loan Purpose",
                ["Education", "Home", "Auto", "Personal"],
                index=["Education", "Home", "Auto", "Personal"].index(def_purp),
                help="Underlying borrowing purpose."
            )

        subcol3, subcol4 = st.columns(2)
        with subcol3:
            loan_amount = st.number_input(
                "Requested Loan Amount (₹)",
                min_value=10000,
                max_value=100000000,
                value=def_loan,
                step=50000,
                help="Total principal amount applied for."
            )
            loan_type = st.selectbox(
                "Loan Type",
                ["Secured", "Unsecured"],
                index=["Secured", "Unsecured"].index(def_type),
                help="Collateral-backed vs unsecured credit facility."
            )
        with subcol4:
            loan_tenure_months = st.number_input(
                "Loan Tenure (Months)",
                min_value=6,
                max_value=360,
                value=def_tenure,
                step=6,
                help="Requested repayment horizon in months."
            )

        # Dynamic LTI Ratio Indicator
        lti_ratio = loan_amount / income if income > 0 else 0
        lti_status = "Safe (< 2.0x)" if lti_ratio < 2.0 else ("Moderate (2.0x - 4.0x)" if lti_ratio <= 4.0 else "High Leverage (> 4.0x)")
        lti_color = "#10b981" if lti_ratio < 2.0 else ("#f59e0b" if lti_ratio <= 4.0 else "#ef4444")
        
        st.markdown(f"""
        <div class="ratio-box">
            <span style="font-size: 0.85rem; color: #64748b; font-weight: 600;">CALCULATED LOAN-TO-INCOME (LTI):</span><br/>
            <span style="font-size: 1.4rem; font-weight: 800; color: {lti_color};">{lti_ratio:.2f}x</span>
            <span style="font-size: 0.85rem; font-weight: 600; color: {lti_color}; margin-left: 0.5rem;">({lti_status})</span>
        </div>
        """, unsafe_allow_html=True)

with col_right:
    st.markdown("### 📊 2. Credit Bureau & Repayment History")
    
    with st.container():
        bcol1, bcol2 = st.columns(2)
        with bcol1:
            credit_utilization_ratio = st.slider(
                "Credit Utilization Ratio (%)",
                min_value=0,
                max_value=100,
                value=def_util,
                step=1,
                help="Revolving credit balance divided by total available credit limit."
            )
            delinquency_ratio = st.slider(
                "Delinquency Ratio (%)",
                min_value=0,
                max_value=100,
                value=def_delinq,
                step=1,
                help="Percentage of past payment cycles with overdue status."
            )
        with bcol2:
            avg_dpd_per_delinquency = st.number_input(
                "Average Days Past Due (DPD)",
                min_value=0,
                max_value=365,
                value=def_dpd,
                step=1,
                help="Average number of days payments remained overdue per delinquency event."
            )
            num_open_accounts = st.number_input(
                "Active Open Accounts",
                min_value=1,
                max_value=10,
                value=def_open_acc,
                step=1,
                help="Number of active active credit facilities reported by the bureau."
            )

        st.caption("💡 *Note: Delinquency Ratio, Avg DPD, and Credit Utilization are primary predictors identified by Information Value (IV) analysis.*")

# ----------------- CALCULATION TRIGGER -----------------
st.markdown("<br/>", unsafe_allow_html=True)
calc_btn = st.button("⚡ Evaluate Credit Risk & Underwriting Decision", type="primary", use_container_width=True)

# ----------------- RESULT SECTION -----------------
if calc_btn:
    with st.spinner("Analyzing applicant risk profile against calibrated scorecard..."):
        probability, credit_score, rating = predict(
            age, income, loan_amount, loan_tenure_months, avg_dpd_per_delinquency,
            delinquency_ratio, credit_utilization_ratio, num_open_accounts,
            residence_type, loan_purpose, loan_type
        )

    # Theme parameters based on rating
    if rating == "Excellent":
        tier_class = "rating-excellent"
        badge_icon = "🟢"
        recommendation = "✅ INSTANT APPROVAL — Prime Borrower. Qualifies for optimal tier-1 interest rates and maximum tenure flexibility."
        decision_box_color = "#10b981"
    elif rating == "Good":
        tier_class = "rating-good"
        badge_icon = "🔵"
        recommendation = "🟢 APPROVE — Low to Moderate Risk. Standard terms and regular underwriting guidelines apply."
        decision_box_color = "#3b82f6"
    elif rating == "Average":
        tier_class = "rating-average"
        badge_icon = "🟡"
        recommendation = "⚠️ CONDITIONAL APPROVAL / MANUAL REVIEW — Moderate Risk. Require secondary guarantor, income verification, or additional collateral."
        decision_box_color = "#f59e0b"
    else:
        tier_class = "rating-poor"
        badge_icon = "🔴"
        recommendation = "❌ DECLINE — High Default Probability. Risk profile exceeds acceptable institutional tolerance threshold."
        decision_box_color = "#ef4444"

    st.markdown("---")
    st.markdown("## 📋 Underwriting Risk Assessment Report")

    # Metrics Row
    m1, m2, m3 = st.columns(3)
    
    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">CALIBRATED CREDIT SCORE</div>
            <div class="metric-value">{credit_score} <span style="font-size: 1rem; color: #94a3b8;">/ 900</span></div>
            <span style="font-size: 0.85rem; color: #64748b; font-weight: 600;">Scale: 300 (High Risk) - 900 (Prime)</span>
        </div>
        """, unsafe_allow_html=True)

    with m2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">DEFAULT PROBABILITY (PD)</div>
            <div class="metric-value" style="color: {decision_box_color};">{probability:.2%}</div>
            <span style="font-size: 0.85rem; color: #64748b; font-weight: 600;">Model Confidence: Macro F1 0.946</span>
        </div>
        """, unsafe_allow_html=True)

    with m3:
        st.markdown(f"""
        <div class="metric-card {tier_class}">
            <div class="metric-title" style="color: inherit;">RISK RATING TIER</div>
            <div class="metric-value" style="color: inherit;">{badge_icon} {rating}</div>
            <span style="font-size: 0.85rem; font-weight: 600;">Category: {rating} Credit Quality</span>
        </div>
        """, unsafe_allow_html=True)

    # Score Gauge Progress Bar
    score_percentage = (credit_score - 300) / 600
    st.progress(max(0.0, min(1.0, score_percentage)))
    col_p1, col_p2, col_p3, col_p4 = st.columns(4)
    col_p1.caption("300 — Poor")
    col_p2.caption("500 — Average")
    col_p3.caption("650 — Good")
    col_p4.caption("750 to 900 — Excellent")

    # Automated Underwriting Directive
    st.markdown(f"""
    <div style="background: rgba(15, 23, 42, 0.04); border-left: 6px solid {decision_box_color}; padding: 1.2rem 1.5rem; border-radius: 8px; margin-top: 1rem; margin-bottom: 1.5rem;">
        <span style="font-size: 0.85rem; font-weight: 700; text-transform: uppercase; color: #64748b;">Underwriting Recommendation Directive:</span>
        <div style="font-size: 1.1rem; font-weight: 700; color: #0f172a; margin-top: 0.3rem;">{recommendation}</div>
    </div>
    """, unsafe_allow_html=True)

    # Detailed Factor Evaluation
    with st.expander("🔍 Detailed Risk Drivers & Feature Breakdown", expanded=True):
        fcol1, fcol2 = st.columns(2)
        with fcol1:
            st.markdown("**🛡️ Positive / Mitigating Factors:**")
            positives = []
            if lti_ratio < 2.5:
                positives.append(f"Low Debt Burden (LTI: {lti_ratio:.2f}x)")
            if credit_utilization_ratio <= 30:
                positives.append(f"Healthy Credit Utilization ({credit_utilization_ratio}%)")
            if delinquency_ratio == 0:
                positives.append("Clean Repayment Record (0% Delinquency)")
            if residence_type == "Owned":
                positives.append("Property Asset Stability (Homeowner)")
            if loan_type == "Secured":
                positives.append("Secured Loan Structure (Collateralized)")
            
            if positives:
                for p in positives:
                    st.markdown(f"- ✅ {p}")
            else:
                st.markdown("*No significant mitigating factors detected.*")

        with fcol2:
            st.markdown("**⚠️ Elevated Risk Factors:**")
            concerns = []
            if lti_ratio >= 3.5:
                concerns.append(f"Elevated Leverage Ratio (LTI: {lti_ratio:.2f}x)")
            if credit_utilization_ratio > 50:
                concerns.append(f"High Credit Utilization ({credit_utilization_ratio}%)")
            if delinquency_ratio > 20:
                concerns.append(f"History of Past Delinquencies ({delinquency_ratio}%)")
            if avg_dpd_per_delinquency >= 30:
                concerns.append(f"High Severity Delinquency (Avg DPD: {avg_dpd_per_delinquency} days)")
            if loan_type == "Unsecured":
                concerns.append("Unsecured Credit Facility")
            
            if concerns:
                for c in concerns:
                    st.markdown(f"- ⚠️ {c}")
            else:
                st.markdown("*No critical risk flags observed.*")

    # Application Audit Trail
    with st.expander("📄 Application Metadata & Audit Trail"):
        st.json({
            "applicant_age": age,
            "annual_income_inr": income,
            "requested_loan_inr": loan_amount,
            "loan_to_income_ratio": round(lti_ratio, 3),
            "loan_tenure_months": loan_tenure_months,
            "residence_type": residence_type,
            "loan_purpose": loan_purpose,
            "loan_type": loan_type,
            "credit_utilization_pct": credit_utilization_ratio,
            "delinquency_ratio_pct": delinquency_ratio,
            "avg_dpd_per_delinquency": avg_dpd_per_delinquency,
            "number_of_open_accounts": num_open_accounts,
            "computed_default_probability": round(probability, 5),
            "calibrated_credit_score": credit_score,
            "risk_rating": rating,
            "model_version": "1.2.0-logistic-optuna"
        })
