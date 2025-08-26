import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Bangladesh Credit Scorer",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS (merged and deduplicated)
st.markdown("""
    <style>
    /* General App Styling */
    .stApp {
        background-color: white !important;
        padding: 0 !important;
        margin: 0 !important;
    }
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 0 !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }
    .main .block-container {
        padding-top: 2rem;
        max-width: 1200px;
    }

    /* Typography */
    h1, h2, h3, p, label, .stMarkdown {
        color: #000000 !important;
    }
    .stMarkdown {
        font-weight: 500;
    }

    /* Metrics */
    div[data-testid="stMetricValue"] {
        color: #000000 !important;
        font-size: 24px;
        font-weight: 600;
    }

    /* Buttons */
    .stButton > button {
        color: white;
        background-color: #1e88e5;
        border-radius: 8px;
        padding: 12px 30px;
        font-weight: 600;
        font-size: 16px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 10px 0;
        width: 100%;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #1565c0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        transform: translateY(-1px);
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background-color: white !important;
    }
    .stTabs [data-baseweb="tab"] {
        font-weight: 500;
    }

    /* Cards */
    .custom-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #eee;
        margin-bottom: 1rem;
    }

    /* Remove spacing */
    .css-1d391kg, .css-1544g2n, .st-emotion-cache-1v0mbdj {
        margin-top: 0 !important;
        padding-top: 0 !important;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("CreditIQ")
st.markdown("##### Advanced Credit Risk Intelligence System")
st.markdown("---")

# Initialize session state
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False

# Load API key
api_key = st.secrets.get("AI_API_KEY", None)
api_choice = st.secrets.get("AI_MODEL", "OpenAI GPT")

# Generate synthetic data
@st.cache_data
def generate_bangladesh_data(n_samples=1000):
    np.random.seed(42)
    data = pd.DataFrame({
        'age': np.random.randint(22, 65, n_samples),
        'monthly_income_bdt': np.random.exponential(35000, n_samples) + 15000,
        'family_members': np.random.poisson(4, n_samples) + 1,
        'education_level': np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.2, 0.25, 0.3, 0.15, 0.1]),
        'division': np.random.choice([1, 2, 3, 4, 5, 6, 7, 8], n_samples),
        'urban_rural': np.random.choice([0, 1], n_samples, p=[0.65, 0.35]),
        'loan_amount_bdt': np.random.exponential(200000, n_samples) + 50000,
        'loan_term_months': np.random.choice([12, 24, 36, 48, 60], n_samples),
        'existing_loans': np.random.poisson(0.5, n_samples),
        'mobile_banking_user': np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
        'bank_account_years': np.random.exponential(3, n_samples),
        'business_type': np.random.choice([1, 2, 3, 4, 5], n_samples),
        'years_in_business': np.random.exponential(4, n_samples),
        'monthly_revenue_bdt': np.random.exponential(80000, n_samples) + 20000,
        'employees': np.random.poisson(2, n_samples),
        'monthly_transactions': np.random.poisson(15, n_samples) + 5,
        'avg_transaction_size_bdt': np.random.exponential(5000, n_samples) + 1000,
        'bank_transaction_sales_ratio': np.random.beta(2, 5, n_samples),
        'previous_loan_default': np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),
        'utility_bill_delays': np.random.poisson(0.3, n_samples),
        'guarantor_available': np.random.choice([0, 1], n_samples, p=[0.4, 0.6]),
        'social_capital_score': np.random.beta(5, 2, n_samples),
        'religious_donations_regular': np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
    })
    default_prob = (
        (data['previous_loan_default'] * 0.3) +
        (data['loan_amount_bdt'] / (data['monthly_income_bdt'] * 12) * 10) +
        (data['utility_bill_delays'] * 0.1) +
        (1 - data['guarantor_available']) * 0.1 +
        (1 - data['mobile_banking_user']) * 0.05 +
        (1 - data['bank_transaction_sales_ratio']) * 0.2 +
        np.random.normal(0, 0.1, n_samples)
    )
    default_prob = 1 / (1 + np.exp(-default_prob))
    data['default'] = (default_prob > 0.5).astype(int)
    return data

# Feature engineering
def create_features(df):
    df = df.copy()
    df['debt_to_income_ratio'] = df['loan_amount_bdt'] / (df['monthly_income_bdt'] * 12)
    df['revenue_per_employee'] = df['monthly_revenue_bdt'] / (df['employees'] + 1)
    df['transaction_frequency_score'] = df['monthly_transactions'] / 30
    df['digital_banking_score'] = df['mobile_banking_user'] * df['bank_account_years']
    df['credit_risk_score'] = (df['previous_loan_default'] * 2 + df['utility_bill_delays']) / 3
    return df

# Load and prepare data
data = generate_bangladesh_data(1500)
data = create_features(data)

feature_columns = [col for col in data.columns if col != 'default']
X = data[feature_columns]
y = data['default']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train models
@st.cache_resource
def train_models():
    lr = LogisticRegression(random_state=42, max_iter=1000).fit(X_train_scaled, y_train)
    rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10).fit(X_train_scaled, y_train)
    gb = GradientBoostingClassifier(n_estimators=100, random_state=42, max_depth=5).fit(X_train_scaled, y_train)
    return lr, rf, gb

# Ensemble prediction
def ensemble_predict(models, X, weights=[0.3, 0.35, 0.35]):
    preds = [model.predict_proba(X)[:, 1] * w for model, w in zip(models, weights)]
    return np.sum(preds, axis=0)

# Credit score and decision
def calculate_credit_score(default_prob):
    return max(1, min(10, round(10 - default_prob * 10, 1)))

def get_decision(score):
    if score >= 7:
        return "APPROVE", "#1a5d1a"
    elif score >= 4:
        return "REVIEW REQUIRED", "#8b4513"
    else:
        return "DECLINE", "#7c0a02"

# Improvement suggestions
def get_improvement_suggestion(feature):
    suggestions = {
        'Monthly Income Bdt': 'Consider additional income sources or business expansion.',
        'Debt To Income Ratio': 'Work on reducing existing debt or increasing income.',
        'Bank Transaction Sales Ratio': 'Increase formal banking transactions.',
        'Years In Business': 'Build longer business operation history.',
        'Bank Account Years': 'Maintain active banking relationships.',
        'Previous Loan Default': 'Clear any outstanding defaults.',
        'Mobile Banking User': 'Adopt digital banking tools.',
        'Credit Risk Score': 'Improve credit history with timely payments.',
        'Monthly Revenue Bdt': 'Focus on increasing business revenue.',
        'Guarantor Available': 'Secure a creditworthy guarantor.',
        'Social Capital Score': 'Build stronger community relationships.',
        'Utility Bill Delays': 'Ensure timely utility bill payments.'
    }
    return suggestions.get(feature, 'Focus on improving this metric.')

# ML explanation
def explain_ml_prediction(models, X_sample, feature_names, scaler):
    lr_model = models[0]
    coefficients = lr_model.coef_[0]
    X_scaled = scaler.transform(X_sample.reshape(1, -1))[0]
    contributions = coefficients * X_scaled
    df = pd.DataFrame({'Feature': feature_names, 'Contribution': contributions})
    df = df.sort_values('Contribution', key=abs, ascending=False)
    pos = df[df['Contribution'] < 0].head(3)
    neg = df[df['Contribution'] > 0].head(3)
    return pos, neg

# AI assessment
def get_ai_assessment(applicant_data, api_key, api_choice, ml_score=None, is_sme=True):
    if not api_key:
        return "AI API key not configured", 5.0, "Missing API key"

    debt_ratio = applicant_data['loan_amount_bdt'] / (applicant_data['monthly_income_bdt'] * 12)
    if is_sme:
        revenue_coverage = applicant_data['monthly_revenue_bdt'] * 12 / applicant_data['loan_amount_bdt']
        business_stability = min(1, applicant_data['years_in_business'] / 5)
        base_score = ((1 - debt_ratio) * 0.3 + (revenue_coverage / 2) * 0.3 +
                      business_stability * 0.2 + (1 - applicant_data['previous_loan_default']) * 0.2) * 10
    else:
        salary_coverage = applicant_data['monthly_income_bdt'] * 12 / applicant_data['loan_amount_bdt']
        base_score = ((1 - debt_ratio) * 0.4 + (salary_coverage / 2) * 0.4 +
                      (1 - applicant_data['previous_loan_default']) * 0.2) * 10

    final_score = (base_score * 0.6 + ml_score * 0.4) if ml_score else base_score
    final_score = max(1, min(10, final_score))
    risk_level = "Low" if final_score >= 7 else "Moderate" if final_score >= 4 else "High"

    strengths = []
    risk_factors = []

    if is_sme:
        if debt_ratio <= 0.5:
            strengths.append(f"Healthy debt service ratio ({debt_ratio:.2f})")
        else:
            risk_factors.append(f"High debt service ratio ({debt_ratio:.2f})")
        if revenue_coverage >= 1.5:
            strengths.append(f"Strong revenue coverage ({revenue_coverage:.2f}x)")
        else:
            risk_factors.append(f"Low revenue coverage ({revenue_coverage:.2f}x)")
        if applicant_data['years_in_business'] >= 3:
            strengths.append(f"Established business ({applicant_data['years_in_business']:.1f} years)")
        else:
            risk_factors.append(f"Limited business history ({applicant_data['years_in_business']:.1f} years)")
    else:
        if debt_ratio <= 0.4:
            strengths.append(f"Manageable debt ratio ({debt_ratio:.2f})")
        else:
            risk_factors.append(f"High debt ratio ({debt_ratio:.2f})")

    if applicant_data['previous_loan_default']:
        risk_factors.append("Previous loan default")
    else:
        strengths.append("Clean credit history")

    strengths_text = "\n".join(f"- {s}" for s in strengths)
    risk_text = "\n".join(f"- {r}" for r in risk_factors) if risk_factors else "No significant risk factors."

    explanation = f"""
Assessment Summary:
Risk Level: {risk_level} (Score: {final_score:.1f}/10)
Strengths:
{strengths_text}
Risk Factors:
{risk_text}
{'Business shows strong potential.' if final_score >= 7 else 'Application shows moderate risk.' if final_score >= 4 else 'Significant risk factors identified.'}
    """.strip()

    return explanation, final_score, f"{'SME' if is_sme else 'Personal'} assessment completed"

# Train or load models
if not st.session_state.model_trained:
    with st.spinner("Training ML models..."):
        lr_model, rf_model, gb_model = train_models()
        st.session_state.update({
            'lr_model': lr_model,
            'rf_model': rf_model,
            'gb_model': gb_model,
            'scaler': scaler,
            'model_trained': True
        })
else:
    lr_model = st.session_state.lr_model
    rf_model = st.session_state.rf_model
    gb_model = st.session_state.gb_model
    scaler = st.session_state.scaler

models = [lr_model, rf_model, gb_model]

# Tabs
tab1, tab2, tab3 = st.tabs(["SME Business", "Employed Businessman", "Benchmark"])

# === SME Tab ===
with tab1:
    st.header("SME Business Loan Assessment")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Business Information")
        business_type = st.selectbox("Business Type", ["Retail Trade", "Manufacturing", "Services", "Agriculture", "Technology"])
        years_in_business = st.number_input("Years in Business", 0.0, 50.0, 3.0)
        monthly_revenue = st.number_input("Monthly Revenue (BDT)", 10000, 10_000_000, 150_000)
        employees = st.number_input("Number of Employees", 0, 500, 5)
        location_type = st.selectbox("Location Type", ["Urban", "Semi-Urban", "Rural"])
        inventory_days = st.number_input("Inventory Turnover Days", 0, 365, 30)

    with col2:
        st.subheader("Financial Details")
        loan_amount = st.number_input("Loan Amount (BDT)", 10000, 50_000_000, 500_000)
        loan_term = st.selectbox("Loan Term (Months)", [12, 24, 36, 48, 60])
        existing_loans = st.number_input("Existing Loans", 0, 10, 0)
        gross_margin = st.slider("Gross Profit Margin (%)", 0, 100, 30)
        working_capital_days = st.number_input("Working Capital Days", 0, 365, 45)

    with col3:
        st.subheader("Credit & Banking")
        bank_account_years = st.number_input("Bank Account Age (Years)", 0.0, 50.0, 2.0)
        mobile_banking = st.checkbox("Mobile Banking User", True)
        previous_default = st.checkbox("Previous Loan Default")
        num_guarantors = st.selectbox("Number of Guarantors", ["0", "1", "2", "3"], index=1)
        if num_guarantors != "0":
            guarantor_types = st.multiselect(
                "Guarantor Type(s)",
                ["Business Owner", "Salaried Professional", "Property Owner", "Government Employee", "Bank Employee", "Corporate Professional"],
                ["Business Owner"]
            )
            guarantor_relationship = st.selectbox("Primary Guarantor Relationship", ["Family Member", "Business Partner", "Professional Associate", "Other"])

    if st.button("Assess Credit Risk", key="sme_assess"):
        monthly_income = 50000  # assumed
        bank_trans_ratio = 0.75

        input_data = pd.DataFrame({
            'age': [35],
            'monthly_income_bdt': [monthly_income],
            'family_members': [4],
            'education_level': [3],
            'division': [1],
            'urban_rural': [1],
            'loan_amount_bdt': [loan_amount],
            'loan_term_months': [loan_term],
            'existing_loans': [existing_loans],
            'mobile_banking_user': [1 if mobile_banking else 0],
            'bank_account_years': [bank_account_years],
            'business_type': [["Retail Trade", "Manufacturing", "Services", "Agriculture", "Technology"].index(business_type) + 1],
            'years_in_business': [years_in_business],
            'monthly_revenue_bdt': [monthly_revenue],
            'employees': [employees],
            'monthly_transactions': [20],
            'avg_transaction_size_bdt': [5000],
            'bank_transaction_sales_ratio': [bank_trans_ratio],
            'previous_loan_default': [1 if previous_default else 0],
            'utility_bill_delays': [0],
            'guarantor_available': [1 if num_guarantors != "0" else 0],
            'social_capital_score': [0.7],
            'religious_donations_regular': [1],
            'debt_to_income_ratio': [loan_amount / (monthly_income * 12)],
            'revenue_per_employee': [monthly_revenue / (employees + 1)],
            'transaction_frequency_score': [0.67],
            'digital_banking_score': [(1 if mobile_banking else 0) * bank_account_years],
            'credit_risk_score': [(1 if previous_default else 0) * 2 / 3]
        })

        input_scaled = scaler.transform(input_data)
        ensemble_prob = ensemble_predict(models, input_scaled)[0]
        ml_score = calculate_credit_score(ensemble_prob)
        decision, color = get_decision(ml_score)

        ai_explanation, ai_score, ai_reasoning = get_ai_assessment(input_data.iloc[0], api_key, api_choice, ml_score, is_sme=True)
        client_score = (ml_score * 0.6 + ai_score * 0.4)

        # Recommendation logic
        recommended_loan = (
            min(loan_amount * 1.2, monthly_revenue * 8) if client_score >= 8 else
            min(loan_amount, monthly_revenue * 6) if client_score >= 6 else
            max(loan_amount * 0.7, monthly_revenue * 4)
        )
        optimal_term = min(loan_term, int(36 * (loan_amount / recommended_loan)))

        loan_message = (
            "💡 Higher loan amount recommended" if client_score >= 8 else
            "✅ Recommended loan amount" if client_score >= 6 else
            "⚠️ Suggesting lower amount"
        )
        loan_color = "#1a5d1a" if client_score >= 8 else "#1565c0" if client_score >= 6 else "#8b4513"

        st.markdown(f"""
        <div style='background: linear-gradient(135deg, {color}22, {color}11); padding: 25px; border-radius: 15px; border: 2px solid {color}33; text-align: center;'>
            <h2 style='color: {color}; margin: 0;'>{decision}</h2>
            <div style='margin: 15px 0; font-size: 16px; color: #555;'>
                {loan_message}<br>
                <strong>Recommended: ৳{recommended_loan:,.0f} over {optimal_term} months</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Analysis toggle
        score_type = st.radio("Analysis Type", ["Combined Analysis", "ML Analysis", "AI Analysis"], horizontal=True)
        if score_type == "Combined Analysis":
            st.markdown(f"""
            <div style='background: {color}22; padding: 25px; border-radius: 15px; text-align: center;'>
                <h3 style='color: {color};'>Combined Score: {client_score:.1f}/10</h3>
                <div>ML: {ml_score}/10 | AI: {ai_score:.1f}/10</div>
                <div>Risk Level: {decision}</div>
            </div>
            """, unsafe_allow_html=True)
        elif score_type == "ML Analysis":
            col1, col2 = st.columns(2)
            with col1:
                st.metric("ML Score", f"{ml_score}/10")
                st.metric("Default Probability", f"{ensemble_prob:.1%}")
            with col2:
                st.metric("Confidence", f"{(1-ensemble_prob)*100:.1f}%")
                st.metric("Decision", decision)
        else:
            st.markdown(f"**AI Score:** {ai_score:.1f}/10")
            st.write(ai_explanation)

        # Improvement areas
        pos_factors, neg_factors = explain_ml_prediction(models, input_data.values[0], feature_columns, scaler)
        st.markdown("### Areas for Improvement")
        for _, row in neg_factors.iterrows():
            feature = row['Feature'].replace('_', ' ').title()
            st.markdown(f"<div style='padding:10px; background:#fff3e0; border-radius:5px; margin:5px 0;'>"
                        f"<strong>{feature}</strong> (Impact: {abs(row['Contribution']):.2f})<br>"
                        f"{get_improvement_suggestion(feature)}</div>", unsafe_allow_html=True)

# === Employed Tab ===
with tab2:
    st.header("Employed Businessman Loan Assessment")
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age", 18, 70, 35, key="emp_age")
        education = st.selectbox("Education", ["Primary", "Secondary", "Higher Secondary", "Bachelor's", "Master's"], key="emp_edu")
        location = st.selectbox("Division", ["Dhaka", "Chittagong", "Rajshahi", "Khulna", "Barisal", "Sylhet", "Rangpur", "Mymensingh"], key="emp_loc")
        location_type = st.selectbox("Location Type", ["Urban", "Semi-Urban", "Rural"], key="emp_loc_type")
        years_at_job = st.number_input("Years at Job", 0.0, 40.0, 2.0, key="emp_job_years")

    with col2:
        employer_type = st.selectbox("Employer Type", ["Government", "Private Company", "Multinational", "NGO", "Self-Employed"], key="emp_type")
        monthly_salary = st.number_input("Monthly Salary (BDT)", 10000, 500000, 60000, key="emp_salary")
        side_income = st.number_input("Side Business Income (BDT)", 0, 500000, 20000, key="emp_side")
        loan_amount_emp = st.number_input("Loan Amount (BDT)", 10000, 10_000_000, 300_000, key="emp_loan")
        loan_term_emp = st.selectbox("Loan Term", [12, 24, 36, 48, 60], key="emp_term")

    with col3:
        bank_years_emp = st.number_input("Banking History (Years)", 0.0, 50.0, 5.0, key="emp_bank")
        mobile_banking_emp = st.checkbox("Mobile Banking User", True, key="emp_mobile")
        previous_default = st.checkbox("Previous Default", key="emp_default")
        num_guarantors = st.selectbox("Number of Guarantors", ["0", "1", "2", "3"], index=1, key="emp_guarantors")
        if num_guarantors != "0":
            st.multiselect("Guarantor Types", ["Salaried Professional", "Government Employee", "Bank Employee", "Corporate Professional"], ["Salaried Professional"], key="emp_gtypes")

    if st.button("Assess Credit Risk", key="emp_assess"):
        total_income = monthly_salary + side_income
        input_data = pd.DataFrame({
            'age': [age],
            'monthly_income_bdt': [total_income],
            'family_members': [4],
            'education_level': [["Primary", "Secondary", "Higher Secondary", "Bachelor's", "Master's"].index(education) + 1],
            'division': [["Dhaka", "Chittagong", "Rajshahi", "Khulna", "Barisal", "Sylhet", "Rangpur", "Mymensingh"].index(location) + 1],
            'urban_rural': [1],
            'loan_amount_bdt': [loan_amount_emp],
            'loan_term_months': [loan_term_emp],
            'existing_loans': [0],
            'mobile_banking_user': [1 if mobile_banking_emp else 0],
            'bank_account_years': [bank_years_emp],
            'business_type': [1],
            'years_in_business': [0],
            'monthly_revenue_bdt': [side_income],
            'employees': [0],
            'monthly_transactions': [25],
            'avg_transaction_size_bdt': [3000],
            'bank_transaction_sales_ratio': [0.8],
            'previous_loan_default': [1 if previous_default else 0],
            'utility_bill_delays': [0],
            'guarantor_available': [1 if num_guarantors != "0" else 0],
            'social_capital_score': [0.8],
            'religious_donations_regular': [1],
            'debt_to_income_ratio': [loan_amount_emp / (total_income * 12)],
            'revenue_per_employee': [side_income],
            'transaction_frequency_score': [0.83],
            'digital_banking_score': [(1 if mobile_banking_emp else 0) * bank_years_emp],
            'credit_risk_score': [(1 if previous_default else 0) * 2 / 3]
        })

        input_scaled = scaler.transform(input_data)
        prob = ensemble_predict(models, input_scaled)[0]
        ml_score = calculate_credit_score(prob)
        decision, color = get_decision(ml_score)

        ai_explanation, ai_score, _ = get_ai_assessment(input_data.iloc[0], api_key, api_choice, ml_score, is_sme=False)

        st.markdown("### Results")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("ML Score", f"{ml_score}/10")
            st.metric("Default Probability", f"{prob:.1%}")
            st.metric("Decision", decision)
        with col2:
            st.metric("AI Score", f"{ai_score:.1f}/10")
            st.write(ai_explanation)

# === Benchmark Tab ===
with tab3:
    st.header("Model Benchmarks")
    view = st.radio("View", ["Model Performance", "Feature Importance"], horizontal=True)
    if view == "Model Performance":
        preds = [m.predict_proba(X_test_scaled)[:, 1] for m in models]
        aucs = [roc_auc_score(y_test, p) for p in preds]
        ensemble_pred = ensemble_predict(models, X_test_scaled)
        ensemble_auc = roc_auc_score(y_test, ensemble_pred)
        st.metric("Ensemble AUC", f"{ensemble_auc:.4f}")
    elif view == "Feature Importance":
        importances = pd.DataFrame({
            'Feature': feature_columns,
            'Importance': rf_model.feature_importances_
        }).sort_values('Importance', ascending=False).head(10)
        for _, row in importances.iterrows():
            st.progress(int(row['Importance'] * 100))
            st.write(f"{row['Feature'].replace('_', ' ').title()}: {row['Importance']:.4f}")

# Footer
st.markdown("---")
st.caption("Credit Risk Assessment System - Bangladesh Context | ML Ensemble & AI Integration")
