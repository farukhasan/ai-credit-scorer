import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, classification_report, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Bangladesh Credit Scorer",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Add spacing for title
st.write('<style>div.block-container{padding-top:1rem;}</style>', unsafe_allow_html=True)

# Custom CSS for spacing and design
st.write('''
    <style>
        div.block-container {padding-top:2rem;}
        div[data-testid="stSidebarContent"] {padding-top: 3rem;}
        div.st-emotion-cache-1v0mbdj.e115fcil1 {margin-top: 1.5rem;}
        .st-emotion-cache-pkbazv {row-gap: 1em;}
        .stMarkdown {padding-top: 0.5rem;}
        .css-1544g2n {margin-top: 0;}
    </style>
''', unsafe_allow_html=True)

# Custom CSS for white background and minimal design
st.markdown("""
    <style>
    .stApp {
        background-color: white !important;
        padding: 0 !important;
        margin: 0 !important;
    }
    .css-1d391kg {
        background-color: white !important;
        padding: 0 !important;
        margin: 0 !important;
    }
    .main {
        margin-top: -4rem !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        background-color: white !important;
        padding: 0 !important;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: white !important;
        padding: 0 !important;
    }
    /* Remove all unnecessary padding */
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 0 !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }
    /* Remove white space between components */
    .element-container {
        margin: 0 !important;
        padding: 0 !important;
    }
    h1, h2, h3, p, label, .stMarkdown {
        color: #000000 !important;
    }
    div[data-testid="stMetricValue"] {
        color: #000000 !important;
    }
    .stMarkdown {
        font-weight: 500;
    }
    /* Button styling */
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
    /* Container styling */
    div.element-container {
        padding-bottom: 1rem;
    }
    /* Remove black spaces */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 0;
        max-width: 1200px;
    }
    /* Metric styling */
    div[data-testid="stMetricValue"] {
        font-size: 24px;
        font-weight: 600;
    }
    /* Tab styling */
    .stTabs [data-baseweb="tab"] {
        font-weight: 500;
    }
    /* Card styling */
    div.element-container div.stMarkdown {
        background-color: transparent;
        padding: 0;
        margin: 0;
    }
    /* Custom card class */
    .custom-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #eee;
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Title and subtitle
st.title("CreditIQ")
st.markdown("##### Advanced Credit Risk Intelligence System")
st.markdown("---")

# Initialize session state
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False

# Use API key from secrets
api_key = st.secrets["AI_API_KEY"]
api_choice = st.secrets.get("AI_MODEL", "OpenAI GPT")

# Generate synthetic Bangladesh context data
@st.cache_data
def generate_bangladesh_data(n_samples=1000):
    np.random.seed(42)
    
    data = pd.DataFrame({
        # Demographic features (Bangladesh context)
        'age': np.random.randint(22, 65, n_samples),
        'monthly_income_bdt': np.random.exponential(35000, n_samples) + 15000,
        'family_members': np.random.poisson(4, n_samples) + 1,
        'education_level': np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.2, 0.25, 0.3, 0.15, 0.1]),
        
        # Location (Division)
        'division': np.random.choice([1, 2, 3, 4, 5, 6, 7, 8], n_samples),
        'urban_rural': np.random.choice([0, 1], n_samples, p=[0.65, 0.35]),
        
        # Financial features
        'loan_amount_bdt': np.random.exponential(200000, n_samples) + 50000,
        'loan_term_months': np.random.choice([12, 24, 36, 48, 60], n_samples),
        'existing_loans': np.random.poisson(0.5, n_samples),
        'mobile_banking_user': np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
        'bank_account_years': np.random.exponential(3, n_samples),
        
        # Business/Employment features
        'business_type': np.random.choice([1, 2, 3, 4, 5], n_samples),
        'years_in_business': np.random.exponential(4, n_samples),
        'monthly_revenue_bdt': np.random.exponential(80000, n_samples) + 20000,
        'employees': np.random.poisson(2, n_samples),
        
        # Transaction features
        'monthly_transactions': np.random.poisson(15, n_samples) + 5,
        'avg_transaction_size_bdt': np.random.exponential(5000, n_samples) + 1000,
        'bank_transaction_sales_ratio': np.random.beta(2, 5, n_samples),
        
        # Credit history
        'previous_loan_default': np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),
        'utility_bill_delays': np.random.poisson(0.3, n_samples),
        
        # Social features (Bangladesh specific)
        'guarantor_available': np.random.choice([0, 1], n_samples, p=[0.4, 0.6]),
        'social_capital_score': np.random.beta(5, 2, n_samples),
        'religious_donations_regular': np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
    })
    
    # Create target variable with logical relationships
    default_prob = (
        (data['previous_loan_default'] * 0.3) +
        (data['loan_amount_bdt'] / data['monthly_income_bdt'] / 100) +
        (data['utility_bill_delays'] * 0.1) +
        (1 - data['guarantor_available']) * 0.1 +
        (1 - data['mobile_banking_user']) * 0.05 +
        (1 - data['bank_transaction_sales_ratio']) * 0.2 +
        np.random.normal(0, 0.1, n_samples)
    )
    
    default_prob = 1 / (1 + np.exp(-default_prob))
    data['default'] = (default_prob > 0.5).astype(int)
    
    return data

# Load or generate data
data = generate_bangladesh_data(1500)

# Feature engineering
def create_features(df):
    df = df.copy()
    df['debt_to_income_ratio'] = df['loan_amount_bdt'] / (df['monthly_income_bdt'] * 12)
    df['revenue_per_employee'] = df['monthly_revenue_bdt'] / (df['employees'] + 1)
    df['transaction_frequency_score'] = df['monthly_transactions'] / 30
    df['digital_banking_score'] = df['mobile_banking_user'] * df['bank_account_years']
    df['credit_risk_score'] = (df['previous_loan_default'] * 2 + df['utility_bill_delays']) / 3
    return df

# Prepare data
data = create_features(data)

# Define features and target
feature_columns = [
    'age', 'monthly_income_bdt', 'family_members', 'education_level',
    'division', 'urban_rural', 'loan_amount_bdt', 'loan_term_months',
    'existing_loans', 'mobile_banking_user', 'bank_account_years',
    'business_type', 'years_in_business', 'monthly_revenue_bdt',
    'employees', 'monthly_transactions', 'avg_transaction_size_bdt',
    'bank_transaction_sales_ratio', 'previous_loan_default',
    'utility_bill_delays', 'guarantor_available', 'social_capital_score',
    'religious_donations_regular', 'debt_to_income_ratio',
    'revenue_per_employee', 'transaction_frequency_score',
    'digital_banking_score', 'credit_risk_score'
]

X = data[feature_columns]
y = data['default']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train models
@st.cache_resource
def train_models():
    # Model 1: Logistic Regression
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train_scaled, y_train)
    
    # Model 2: Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    rf_model.fit(X_train_scaled, y_train)
    
    # Model 3: Gradient Boosting
    gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42, max_depth=5)
    gb_model.fit(X_train_scaled, y_train)
    
    return lr_model, rf_model, gb_model

# Ensemble prediction
def ensemble_predict(models, X, weights=[0.3, 0.35, 0.35]):
    predictions = []
    for model, weight in zip(models, weights):
        pred_proba = model.predict_proba(X)[:, 1]
        predictions.append(pred_proba * weight)
    
    ensemble_pred = np.sum(predictions, axis=0)
    return ensemble_pred

# Calculate credit score (1-10)
def calculate_credit_score(default_prob):
    score = 10 - (default_prob * 10)
    return max(1, min(10, round(score, 1)))

# Get decision
def get_decision(score):
    if score >= 7:
        return "APPROVE", "#1a5d1a"  # Dark green
    elif score >= 4:
        return "REVIEW REQUIRED", "#8b4513"  # Saddle brown
    else:
        return "DECLINE", "#7c0a02"  # Dark red

# Get improvement suggestions
def get_improvement_suggestion(feature):
    suggestions = {
        'Monthly Income Bdt': 'Consider additional income sources or business expansion opportunities',
        'Debt To Income Ratio': 'Work on reducing existing debt or increasing income to improve ratio',
        'Bank Transaction Sales Ratio': 'Increase formal banking transactions for business operations',
        'Years In Business': 'Build longer business operation history and maintain proper records',
        'Bank Account Years': 'Maintain active banking relationships and regular transactions',
        'Previous Loan Default': 'Clear any outstanding defaults and maintain timely payments',
        'Mobile Banking User': 'Adopt digital banking for better transaction tracking',
        'Credit Risk Score': 'Improve credit history through timely bill payments',
        'Monthly Revenue Bdt': 'Focus on increasing business revenue and maintaining records',
        'Guarantor Available': 'Secure a creditworthy guarantor for the loan',
        'Social Capital Score': 'Build stronger business relationships in the community',
        'Utility Bill Delays': 'Ensure timely payment of all utility bills'
    }
    return suggestions.get(feature, 'Focus on improving this metric based on industry standards')

# ML Explainer
def explain_ml_prediction(models, X_sample, feature_names, scaler):
    # Get feature contributions (simplified SHAP-like approach)
    lr_model = models[0]
    coefficients = lr_model.coef_[0]
    
    X_scaled = scaler.transform(X_sample.reshape(1, -1))[0]
    contributions = coefficients * X_scaled
    
    # Get top positive and negative factors
    factor_df = pd.DataFrame({
        'Feature': feature_names,
        'Contribution': contributions
    })
    
    factor_df = factor_df.sort_values('Contribution', key=abs, ascending=False)
    
    positive_factors = factor_df[factor_df['Contribution'] < 0].head(3)  # Lower contribution = lower default risk
    negative_factors = factor_df[factor_df['Contribution'] > 0].head(3)  # Higher contribution = higher default risk
    
    return positive_factors, negative_factors

# AI Risk Assessment
def get_ai_assessment(applicant_data, api_key, api_choice, ml_score=None, is_sme=True):
    if not api_key:
        return "Please provide API key for AI assessment", 5, "AI API key not configured"
    
    # Initialize explanation components
    risk_factors = []
    strengths = []
    
    # Analyze application data
    if is_sme:
        factors = {
            'revenue_ratio': applicant_data['monthly_revenue_bdt'] * 12 / applicant_data['loan_amount_bdt'],
            'debt_service': applicant_data['loan_amount_bdt'] / (applicant_data['monthly_revenue_bdt'] * 12),
            'business_age': applicant_data['years_in_business'],
            'credit_history': not applicant_data['previous_loan_default']
        }
    else:
        factors = {
            'income_ratio': applicant_data['monthly_income_bdt'] * 12 / applicant_data['loan_amount_bdt'],
            'debt_service': applicant_data['loan_amount_bdt'] / (applicant_data['monthly_income_bdt'] * 12),
            'banking_score': applicant_data['bank_account_years'] * (1 if applicant_data['mobile_banking_user'] else 0.8),
            'credit_history': not applicant_data['previous_loan_default']
        }
    
    # Calculate base risk score using key financial ratios
    debt_service_ratio = applicant_data['loan_amount_bdt'] / (applicant_data['monthly_income_bdt'] * 12)
    if is_sme:
        revenue_coverage = applicant_data['monthly_revenue_bdt'] * 12 / applicant_data['loan_amount_bdt']
        business_stability = min(1, applicant_data['years_in_business'] / 5)
        base_score = (
            (1 - debt_service_ratio) * 0.3 +
            (revenue_coverage/2) * 0.3 +
            business_stability * 0.2 +
            (1 - applicant_data['previous_loan_default']) * 0.2
        ) * 10
    else:
        salary_coverage = applicant_data['monthly_income_bdt'] * 12 / applicant_data['loan_amount_bdt']
        base_score = (
            (1 - debt_service_ratio) * 0.4 +
            (salary_coverage/2) * 0.4 +
            (1 - applicant_data['previous_loan_default']) * 0.2
        ) * 10

    # Incorporate ML score if available
    if ml_score is not None:
        final_score = (base_score * 0.6 + ml_score * 0.4)
    else:
        final_score = base_score
        
    final_score = max(1, min(10, final_score))
    
    # Generate detailed analysis
    factors = []
    if is_sme:
        if debt_service_ratio > 0.5:
            factors.append(f"High debt service ratio ({debt_service_ratio:.2f})")
        if revenue_coverage < 1.5:
            factors.append(f"Low revenue coverage ({revenue_coverage:.2f}x)")
        if applicant_data['years_in_business'] < 3:
            factors.append(f"Limited business history ({applicant_data['years_in_business']:.1f} years)")
    else:
        if debt_service_ratio > 0.4:
            factors.append(f"High debt service ratio ({debt_service_ratio:.2f})")
        if salary_coverage < 2:
            factors.append(f"Low salary coverage ({salary_coverage:.2f}x)")
    
    if applicant_data['previous_loan_default']:
        factors.append("Previous default history")
    
    # Risk assessment notes for future enhancement:
    # - Consider Bangladesh's economic context
    # - Rate credit risk from 1-10 (10 being lowest risk)
    # - Explain why applicant might default in 2-3 sentences
    
    # Analyze risk factors
    if is_sme:
        if debt_service_ratio > 0.5:
            risk_factors.append(f"Debt service ratio ({debt_service_ratio:.2f}) above recommended level")
        else:
            strengths.append(f"Healthy debt service ratio of {debt_service_ratio:.2f}")

        if revenue_coverage < 1.5:
            risk_factors.append(f"Low revenue coverage ({revenue_coverage:.2f}x)")
        else:
            strengths.append(f"Strong revenue coverage of {revenue_coverage:.2f}x")

        if business_stability < 0.6:
            risk_factors.append(f"Limited business history ({applicant_data['years_in_business']:.1f} years)")
        else:
            strengths.append(f"Established business ({applicant_data['years_in_business']:.1f} years)")
    else:
        if debt_service_ratio > 0.4:
            risk_factors.append(f"High debt service ratio ({debt_service_ratio:.2f})")
        else:
            strengths.append(f"Manageable debt service ratio of {debt_service_ratio:.2f}")

    if applicant_data['previous_loan_default']:
        risk_factors.append("Previous default history")
    else:
        strengths.append("Clean credit history")

    # Calculate final score
    if ml_score is not None:
        final_score = base_score * 0.6 + ml_score * 0.4
    else:
        final_score = base_score

    final_score = max(1, min(10, final_score))
    risk_level = "Low" if final_score >= 7 else "Moderate" if final_score >= 4 else "High"

    # Generate explanation
    strengths_text = "\n".join('- ' + s for s in strengths)
    risk_text = "\n".join('- ' + r for r in risk_factors) if risk_factors else 'No significant risk factors identified.'
    
    assessment_result = (
        'strong potential with manageable risk levels.' if final_score >= 7
        else 'moderate risk with some concerns that need attention.' if final_score >= 4
        else 'significant risk factors that require careful consideration.'
    )
    
    explanation = (
        f"Assessment Summary for {'SME' if is_sme else 'Salaried'} Applicant:\n"
        f"Risk Level: {risk_level} (Score: {final_score:.1f}/10)\n\n"
        f"Strengths:\n{strengths_text}\n\n"
        f"Risk Factors:\n{risk_text}\n\n"
        f"{'Business' if is_sme else 'Application'} shows {assessment_result}"
    )

    return explanation, final_score, f"{'Business' if is_sme else 'Personal'} risk assessment completed"
    
    business_context = "SME business" if is_sme else "Salaried professional"
    
    explanation = f"""Based on comprehensive analysis of this {business_context} application in Bangladesh:
    
    Risk Level: {risk_level} (Score: {final_score:.1f}/10)
    
    {risk_factors}
    
    {'Business shows ' if is_sme else 'Application shows '}
    {
        'strong potential with manageable risk levels.' if final_score >= 7 else
        'moderate risk with some concerns that need attention.' if final_score >= 4 else
        'significant risk factors that require careful consideration.'
    }
    
    {'Business stability and revenue generation are key strengths.' if is_sme and final_score >= 7 else
     'Stable employment and income levels support the application.' if not is_sme and final_score >= 7 else
     'Additional risk mitigation measures are recommended.' if final_score >= 4 else
     'Significant improvements needed in financial metrics.'}
    """
    
    return (
        explanation,
        final_score,
        f"{'Business' if is_sme else 'Personal'} risk assessment completed"
    )

# Main Application
tab1, tab2, tab3 = st.tabs(["SME Business", "Employed Businessman", "Benchmark"])

# Train models
if not st.session_state.model_trained:
    with st.spinner("Training ML models..."):
        lr_model, rf_model, gb_model = train_models()
        st.session_state.lr_model = lr_model
        st.session_state.rf_model = rf_model
        st.session_state.gb_model = gb_model
        st.session_state.model_trained = True
        st.session_state.scaler = scaler
else:
    lr_model = st.session_state.lr_model
    rf_model = st.session_state.rf_model
    gb_model = st.session_state.gb_model
    scaler = st.session_state.scaler

models = [lr_model, rf_model, gb_model]

with tab1:
    st.header("SME Business Loan Assessment")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Business Information")
        business_type = st.selectbox("Business Type", 
            ["Retail Trade", "Manufacturing", "Services", "Agriculture", "Technology"])
        years_in_business = st.number_input("Years in Business", min_value=0.0, max_value=50.0, value=3.0)
        monthly_revenue = st.number_input("Monthly Revenue (BDT)", min_value=10000, max_value=10000000, value=150000)
        employees = st.number_input("Number of Employees", min_value=0, max_value=500, value=5)
        location_type = st.selectbox("Location Type", ["Urban", "Semi-Urban", "Rural"])
        inventory_days = st.number_input("Inventory Turnover Days", min_value=0, max_value=365, value=30)
        
    with col2:
        st.subheader("Financial Details")
        loan_amount = st.number_input("Loan Amount (BDT)", min_value=10000, max_value=50000000, value=500000)
        loan_term = st.selectbox("Loan Term (Months)", [12, 24, 36, 48, 60])
        existing_loans = st.number_input("Existing Loans", min_value=0, max_value=10, value=0)
        gross_margin = st.slider("Gross Profit Margin (%)", 0, 100, 30)
        working_capital_days = st.number_input("Working Capital Days", min_value=0, max_value=365, value=45)
        
    with col3:
        st.subheader("Credit & Banking")
        bank_account_years = st.number_input("Bank Account Age (Years)", min_value=0.0, max_value=50.0, value=2.0)
        mobile_banking = st.checkbox("Mobile Banking User", value=True)
        previous_default = st.checkbox("Previous Loan Default")
        num_guarantors = st.selectbox("Number of Guarantors", [1, 2, 3])
        guarantor_types = st.multiselect(
            "Guarantor Types",
            ["Business Owner", "Salaried Professional", "Property Owner", "Government Employee"],
            ["Business Owner"]
        )
        st.markdown("""
            <div style='background-color: #f8f9fa; padding: 20px; border-radius: 10px; margin-top: 20px;'>
                <h4 style='margin: 0 0 15px 0; color: #2c3e50;'>Guarantor Information</h4>
            </div>
        """, unsafe_allow_html=True)
        
        num_guarantors = st.number_input("Number of Guarantors", min_value=0, max_value=3, value=1)
        if num_guarantors > 0:
            with st.container():
                st.markdown("""
                    <div style='background-color: white; padding: 15px; border-radius: 8px; border: 1px solid #e0e0e0;'>
                """, unsafe_allow_html=True)
                
                guarantor_types = st.multiselect(
                    "Guarantor Type",
                    ["Business Owner", "Salaried Professional", "Property Owner", "Government Employee", 
                     "Bank Employee", "Corporate Professional", "Others"],
                    ["Business Owner"]
                )
                
                col1, col2 = st.columns(2)
                with col1:
                    guarantor_relationship = st.selectbox(
                        "Primary Guarantor Relationship",
                        ["Family Member", "Business Partner", "Professional Associate", "Other"]
                    )
                
                with col2:
                    if "Others" in guarantor_types:
                        other_guarantor_type = st.text_input("Specify Other Guarantor Type")
                
                st.markdown("</div>", unsafe_allow_html=True)
    
    if st.button("Assess Credit Risk", key="sme_assess"):
        # Prepare input data
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
            'guarantor_available': [1 if num_guarantors > 0 else 0],
            'social_capital_score': [0.7],
            'religious_donations_regular': [1],
            'debt_to_income_ratio': [loan_amount / (monthly_income * 12)],
            'revenue_per_employee': [monthly_revenue / (employees + 1)],
            'transaction_frequency_score': [0.67],
            'digital_banking_score': [(1 if mobile_banking else 0) * bank_account_years],
            'credit_risk_score': [(1 if previous_default else 0) * 2 / 3]
        })
        
        # Scale input
        input_scaled = scaler.transform(input_data)
        
        # Get predictions
        ensemble_prob = ensemble_predict(models, input_scaled)[0]
        ml_score = calculate_credit_score(ensemble_prob)
        decision, color = get_decision(ml_score)
        
        # Display results in a structured format
        st.markdown("---")
        
        # Get AI assessment first
        ai_explanation, ai_score, ai_reasoning = get_ai_assessment(input_data.iloc[0], api_key, api_choice)
        
        # Enhanced Dynamic Recommendations Engine
        client_score = (ml_score * 0.6 + ai_score * 0.4)  # Combined score
        max_possible_loan = monthly_revenue * 8  # 8 months revenue for excellent clients
        min_suggested_loan = monthly_revenue * 4  # 4 months revenue for risky clients
        optimal_loan = min(monthly_revenue * 6, loan_amount)  # 6 months revenue as baseline
        
        if client_score >= 8:
            recommended_loan = min(loan_amount * 1.2, max_possible_loan)  # Can offer 20% more
            loan_message = "💡 Based on your excellent profile, you qualify for a higher loan amount"
            loan_color = "#1a5d1a"  # Dark green
        elif client_score >= 6:
            recommended_loan = min(loan_amount, optimal_loan)  # Based on revenue
            loan_message = "✓ Recommended loan amount based on business revenue"
            loan_color = "#1565c0"  # Blue
        else:
            recommended_loan = min(loan_amount * 0.7, min_suggested_loan)  # 30% reduction
            loan_message = "⚠️ Suggesting a lower amount to improve approval chances"
            loan_color = "#8b4513"  # Brown
            
        optimal_term = min(loan_term, int(36 * (loan_amount / recommended_loan)))
        
        # Modern Decision Card
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, {color}22, {color}11);
             padding: 25px; border-radius: 15px; border: 2px solid {color}33;
             margin-bottom: 30px; text-align: center;'>
            <h2 style='color: {color}; margin: 0; font-size: 28px;'>{decision}</h2>
            <div style='margin: 15px 0; font-size: 16px; color: #555;'>
                {loan_message}<br/>
                Recommended Amount: ৳{recommended_loan:,.0f} over {optimal_term} months
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Get AI assessment
        ai_explanation, ai_score, ai_reasoning = get_ai_assessment(input_data.iloc[0], api_key, api_choice)
        
        # Calculate combined score
        client_score = (ml_score * 0.6 + ai_score * 0.4)
        
        # Improved Scoring Display with Toggle
        st.markdown("### Credit Assessment Analysis")
        score_type = st.radio("Select Analysis Type", ["Combined Analysis", "ML Analysis", "AI Analysis"], horizontal=True)
        
        if score_type == "Combined Analysis":
            col1, col2, col3 = st.columns([1,2,1])
            with col2:
                st.markdown(f"""
                <div style='background: linear-gradient(135deg, {color}22, {color}11);
                     padding: 25px; border-radius: 15px; border: 2px solid {color}33; text-align: center;'>
                    <h3 style='margin: 0; color: {color};'>Combined Score: {client_score:.1f}/10</h3>
                    <div style='margin: 15px 0; color: #555;'>
                        ML Score: {ml_score}/10 (60%) | AI Score: {ai_score:.1f}/10 (40%)
                    </div>
                    <div style='font-size: 14px; color: #666;'>
                        Risk Level: {decision}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        elif score_type == "ML Analysis":
            st.markdown("""
            <div style='background-color: #f8f9fa; padding: 20px; border-radius: 10px; border: 1px solid #e0e0e0;'>
                <h4 style='margin: 0 0 15px 0; color: #2c3e50;'>Machine Learning Detailed Analysis</h4>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("ML Credit Score", f"{ml_score}/10", "Primary Score")
                st.metric("Default Probability", f"{ensemble_prob:.1%}", "Risk Assessment")
            with col2:
                st.metric("Model Confidence", f"{(1-ensemble_prob)*100:.1f}%", "Approval Confidence")
                st.metric("Risk Category", decision)
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        else:  # AI Analysis
            st.markdown("""
            <div style='background-color: #f8f9fa; padding: 20px; border-radius: 10px; border: 1px solid #e0e0e0;'>
                <h4 style='margin: 0 0 15px 0; color: #2c3e50;'>AI Contextual Analysis</h4>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("AI Credit Score", f"{ai_score:.1f}/10", "Contextual Score")
                st.metric("Risk Category", ai_reasoning)
            with col2:
                st.markdown("#### AI Insights")
                st.write(ai_explanation)
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Enhanced Risk Analysis Cards
        st.markdown("### Comprehensive Risk Analysis")
        analysis_type = st.radio("Analysis View", ["Risk Factors", "Business Health", "Financial Metrics"], horizontal=True)
        
        if analysis_type == "Risk Factors":
            # Calculated Risk Components
            risk_components = {
                'Business Stability': {
                    'score': min(10, years_in_business * 2),
                    'weight': 0.3,
                    'status': 'Strong' if years_in_business >= 3 else 'Growing',
                    'detail': f'Business age: {years_in_business:.1f} years'
                },
                'Financial Health': {
                    'score': min(10, (monthly_revenue / loan_amount) * 5),
                    'weight': 0.25,
                    'status': 'Healthy' if monthly_revenue > loan_amount/24 else 'Moderate',
                    'detail': f'Revenue to loan ratio: {(monthly_revenue*12/loan_amount):.1f}x'
                },
                'Credit Profile': {
                    'score': 10 if not previous_default else 3,
                    'weight': 0.25,
                    'status': 'Excellent' if not previous_default else 'Needs Attention',
                    'detail': 'No defaults' if not previous_default else 'Has previous defaults'
                },
                'Banking Profile': {
                    'score': min(10, bank_account_years * 2),
                    'weight': 0.2,
                    'status': 'Established' if bank_account_years >= 3 else 'Developing',
                    'detail': f'Banking history: {bank_account_years:.1f} years'
                }
            }
            
            for component, details in risk_components.items():
                score = details['score'] * details['weight']
                st.markdown(f"""
                <div style='padding: 10px; border-radius: 5px; background-color: #ffffff; border: 1px solid #e0e0e0; margin: 5px 0;'>
                    <div style='display: flex; justify-content: space-between; align-items: center;'>
                        <strong>{component}</strong>
                        <span style='color: {"#1a5d1a" if details["status"] == "Good" else "#8b4513"}'>
                            {details["status"]}
                        </span>
                    </div>
                    <div style='background-color: #e3f2fd; width: {details["score"]*10}%; height: 6px; border-radius: 3px; margin: 5px 0;'></div>
                    <small>Contribution: {score:.1f} points</small>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            # Strengths & Weaknesses
            positive_factors, negative_factors = explain_ml_prediction(
                models, input_data.values[0], feature_columns, scaler
            )
            
            st.markdown("""
            <div style='padding: 15px; border-radius: 5px; background-color: #ffffff; border: 1px solid #e0e0e0;'>
                <h4>Key Strengths</h4>
            </div>
            """, unsafe_allow_html=True)
            
            for _, row in positive_factors.iterrows():
                feature = row['Feature'].replace('_', ' ').title()
                st.markdown(f"""
                <div style='padding: 10px; margin: 5px 0; background-color: #e8f5e9; border-radius: 5px;'>
                    <strong>{feature}</strong><br/>
                    Impact: {abs(row['Contribution']):.2f} points
                </div>
                """, unsafe_allow_html=True)
        
        # Improvement Areas Card
        st.markdown("### Areas for Improvement")
        for _, row in negative_factors.iterrows():
            feature = row['Feature'].replace('_', ' ').title()
            improvement = get_improvement_suggestion(feature)
            st.markdown(f"""
            <div style='padding: 15px; margin: 10px 0; background-color: #fff3e0; border-radius: 5px;'>
                <strong>{feature}</strong> (Risk Impact: {abs(row['Contribution']):.2f} points)<br/>
                {improvement}
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("### AI Assessment")
            ai_explanation, ai_score, ai_reasoning = get_ai_assessment(input_data.iloc[0], api_key, api_choice)
            st.metric("AI Credit Score", f"{ai_score:.1f}/10")
            st.markdown("#### AI Explanation")
            st.write(ai_explanation)
            st.info(ai_reasoning)

with tab2:
    st.header("Employed Businessman Loan Assessment")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Personal Information")
        age = st.number_input("Age", min_value=18, max_value=70, value=35, key="emp_age")
        education = st.selectbox("Education Level", 
            ["Primary", "Secondary", "Higher Secondary", "Bachelor's", "Master's"], key="emp_edu")
        location = st.selectbox("Division", 
            ["Dhaka", "Chittagong", "Rajshahi", "Khulna", "Barisal", "Sylhet", "Rangpur", "Mymensingh"], key="emp_loc")
        location_type = st.selectbox("Location Type", ["Urban", "Semi-Urban", "Rural"], key="emp_loc_type")
        years_at_job = st.number_input("Years at Current Job", min_value=0.0, max_value=40.0, value=2.0, key="emp_job_years")
        
    with col2:
        st.subheader("Employment & Income")
        employer_type = st.selectbox("Employer Type", 
            ["Government", "Private Company", "Multinational", "NGO", "Self-Employed"], key="emp_type")
        monthly_salary = st.number_input("Monthly Salary (BDT)", min_value=10000, max_value=500000, value=60000, key="emp_salary")
        side_business_income = st.number_input("Side Business Income (BDT)", min_value=0, max_value=500000, value=20000, key="emp_side")
        loan_amount_emp = st.number_input("Loan Amount (BDT)", min_value=10000, max_value=10000000, value=300000, key="emp_loan")
        loan_term_emp = st.selectbox("Loan Term (Months)", [12, 24, 36, 48, 60], key="emp_term")
        
    with col3:
        st.subheader("Credit & Banking")
        bank_years_emp = st.number_input("Banking History (Years)", min_value=0.0, max_value=50.0, value=5.0, key="emp_bank")
        mobile_banking_emp = st.checkbox("Mobile Banking User", value=True, key="emp_mobile")
        previous_default = st.checkbox("Previous Default", key="emp_default")
        num_guarantors = st.selectbox("Number of Guarantors", [1, 2, 3], key="emp_num_guarantors")
        guarantor_types = st.multiselect(
            "Guarantor Types",
            ["Salaried Professional", "Government Employee", "Bank Employee", "Corporate Professional"],
            ["Salaried Professional"],
            key="emp_guarantor_types"
        )
    
    if st.button("Assess Credit Risk", key="emp_assess"):
        # Similar assessment logic as SME tab
        total_income = monthly_salary + side_business_income
        
        input_data_emp = pd.DataFrame({
            'age': [age],
            'monthly_income_bdt': [total_income],
            'family_members': [family_members],
            'education_level': [["Primary", "Secondary", "Higher Secondary", "Bachelor's", "Master's"].index(education) + 1],
            'division': [["Dhaka", "Chittagong", "Rajshahi", "Khulna", "Barisal", "Sylhet", "Rangpur", "Mymensingh"].index(location) + 1],
            'urban_rural': [1],
            'loan_amount_bdt': [loan_amount_emp],
            'loan_term_months': [36],
            'existing_loans': [0],
            'mobile_banking_user': [1 if mobile_banking_emp else 0],
            'bank_account_years': [bank_years_emp],
            'business_type': [1],
            'years_in_business': [0],
            'monthly_revenue_bdt': [side_business_income],
            'employees': [0],
            'monthly_transactions': [25],
            'avg_transaction_size_bdt': [3000],
            'bank_transaction_sales_ratio': [trans_ratio_emp],
            'previous_loan_default': [1 if default_emp else 0],
            'utility_bill_delays': [0],
            'guarantor_available': [1 if guarantor_emp else 0],
            'social_capital_score': [0.8],
            'religious_donations_regular': [1],
            'debt_to_income_ratio': [loan_amount_emp / (total_income * 12)],
            'revenue_per_employee': [side_business_income],
            'transaction_frequency_score': [0.83],
            'digital_banking_score': [(1 if mobile_banking_emp else 0) * bank_years_emp],
            'credit_risk_score': [(1 if default_emp else 0) * 2 / 3]
        })
        
        # Scale and predict
        input_scaled_emp = scaler.transform(input_data_emp)
        ensemble_prob_emp = ensemble_predict(models, input_scaled_emp)[0]
        ml_score_emp = calculate_credit_score(ensemble_prob_emp)
        decision_emp, color_emp = get_decision(ml_score_emp)
        
        # Display results
        st.markdown("---")
        st.subheader("Assessment Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### ML Model Assessment")
            st.metric("Credit Score", f"{ml_score_emp}/10")
            st.metric("Default Probability", f"{ensemble_prob_emp:.2%}")
            st.metric("Decision", decision_emp)
            
            # ML Explanation
            st.markdown("#### Risk Factors Analysis")
            positive_factors_emp, negative_factors_emp = explain_ml_prediction(
                models, input_data_emp.values[0], feature_columns, scaler
            )
            
            st.markdown("**Positive Factors:**")
            for _, row in positive_factors_emp.iterrows():
                st.write(f"• {row['Feature'].replace('_', ' ').title()}")
            
            st.markdown("**Risk Factors:**")
            for _, row in negative_factors_emp.iterrows():
                st.write(f"• {row['Feature'].replace('_', ' ').title()}")
        
        with col2:
            st.markdown("### AI Assessment")
            ai_explanation_emp, ai_score_emp, ai_reasoning_emp = get_ai_assessment(
                input_data_emp.iloc[0], api_key, api_choice
            )
            st.metric("AI Credit Score", f"{ai_score_emp:.1f}/10")
            st.markdown("#### AI Explanation")
            st.write(ai_explanation_emp)
            st.info(ai_reasoning_emp)

# Benchmark tab content
with tab3:
    st.header("Credit Assessment Benchmarks")
    
    # Add benchmark selector
    benchmark_view = st.radio(
        "Select Benchmark View",
        ["Model Performance", "Industry Statistics", "Regional Comparison"],
        horizontal=True
    )
    
    if benchmark_view == "Model Performance":
        # Calculate model performance metrics
        lr_pred = lr_model.predict_proba(X_test_scaled)[:, 1]
        rf_pred = rf_model.predict_proba(X_test_scaled)[:, 1]
        gb_pred = gb_model.predict_proba(X_test_scaled)[:, 1]
        ensemble_pred = ensemble_predict(models, X_test_scaled)
        
        lr_auc = roc_auc_score(y_test, lr_pred)
        rf_auc = roc_auc_score(y_test, rf_pred)
        gb_auc = roc_auc_score(y_test, gb_pred)
        ensemble_auc = roc_auc_score(y_test, ensemble_pred)

        # Display model performance metrics in cards
        st.markdown("### Model Performance Metrics")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div style='padding: 20px; border-radius: 10px; background-color: #f8f9fa; margin-bottom: 20px;'>
                <h4 style='margin-top: 0;'>Individual Model Performance</h4>
            </div>
            """, unsafe_allow_html=True)
            
            metrics = {
                'Logistic Regression': {
                    'AUC': lr_auc,
                    'Accuracy': lr_model.score(X_test_scaled, y_test)
                },
                'Random Forest': {
                    'AUC': rf_auc,
                    'Accuracy': rf_model.score(X_test_scaled, y_test)
                },
                'Gradient Boosting': {
                    'AUC': gb_auc,
                    'Accuracy': gb_model.score(X_test_scaled, y_test)
                }
            }
            
            for model_name, scores in metrics.items():
                st.markdown(f"""
                <div style='padding: 15px; border-radius: 5px; background-color: #ffffff; border: 1px solid #e0e0e0; margin: 10px 0;'>
                    <h5 style='margin: 0;'>{model_name}</h5>
                    <p style='margin: 5px 0;'>AUC Score: {scores['AUC']:.4f}</p>
                    <p style='margin: 5px 0;'>Accuracy: {scores['Accuracy']:.4f}</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style='padding: 20px; border-radius: 10px; background-color: #f8f9fa; margin-bottom: 20px;'>
                <h4 style='margin-top: 0;'>Ensemble Model Performance</h4>
            </div>
            """, unsafe_allow_html=True)
            
            st.metric("Ensemble AUC Score", f"{ensemble_auc:.4f}", "Combined Performance")
            st.metric("Accuracy", f"{((ensemble_pred > 0.5) == y_test).mean():.4f}", "Overall Accuracy")
            st.metric("F1 Score", f"{f1_score(y_test, ensemble_pred > 0.5):.4f}", "Balance Score")
        
        # Feature importance analysis
        st.markdown("### Feature Importance Analysis")
        
        # Get feature importance from Random Forest
        feature_importance = pd.DataFrame({
            'Feature': feature_columns,
            'Importance': rf_model.feature_importances_
        }).sort_values('Importance', ascending=False).head(10)
        
        st.markdown("""
        <div style='padding: 20px; border-radius: 10px; background-color: #f8f9fa; margin: 20px 0;'>
            <h4 style='margin-top: 0;'>Top 10 Most Important Features</h4>
        </div>
        """, unsafe_allow_html=True)
        
        for _, row in feature_importance.iterrows():
            st.markdown(f"""
            <div style='padding: 10px; border-radius: 5px; background-color: #ffffff; border: 1px solid #e0e0e0; margin: 5px 0;'>
                <strong>{row['Feature'].replace('_', ' ').title()}</strong>
                <div style='background-color: #e3f2fd; width: {row['Importance']*100:.1f}%; height: 10px; border-radius: 5px;'></div>
                <small>Importance: {row['Importance']:.4f}</small>
            </div>
            """, unsafe_allow_html=True)

    elif benchmark_view == "Industry Statistics":
        st.subheader("Industry Benchmarks")
        # Add industry statistics comparison
        # Add code for industry statistics view

    elif benchmark_view == "Regional Comparison":
        st.subheader("Regional Performance")
        # Add regional comparison metrics
        # Add code for regional comparison view

# Footer
st.markdown("---")
st.caption("Credit Risk Assessment System - Bangladesh Context | Using ML Ensemble & AI Integration")
st.caption("Models: Logistic Regression + Random Forest + Gradient Boosting | AI Integration")
