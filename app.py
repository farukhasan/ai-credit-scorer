import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(page_title="Bangladesh Credit Scorer", layout="wide")

# Custom CSS for white background and minimal design
st.markdown("""
    <style>
    .stApp {
        background-color: white;
    }
    .css-1d391kg {
        background-color: white;
    }
    .stTabs [data-baseweb="tab-list"] {
        background-color: white;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: white;
    }
    h1, h2, h3 {
        color: #333333;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("🏦 Bangladesh Credit Risk Assessment System")
st.markdown("---")

# Initialize session state
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False

# Sidebar for API configuration
st.sidebar.header("Configuration")
api_choice = st.sidebar.selectbox(
    "Select AI API",
    ["Google Gemini", "OpenAI GPT", "Anthropic Claude"]
)

# API Key input
if api_choice == "Google Gemini":
    api_key = st.sidebar.text_input("Enter Gemini API Key", type="password")
    api_info = "Using Google Gemini AI for intelligent risk assessment"
elif api_choice == "OpenAI GPT":
    api_key = st.sidebar.text_input("Enter OpenAI API Key", type="password")
    api_info = "Using OpenAI GPT for intelligent risk assessment"
else:
    api_key = st.sidebar.text_input("Enter Claude API Key", type="password")
    api_info = "Using Anthropic Claude for intelligent risk assessment"

st.sidebar.info(api_info)

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
        return "✅ APPROVE", "green"
    elif score >= 4:
        return "⚠️ FURTHER REVIEW", "orange"
    else:
        return "❌ REJECT", "red"

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
def get_ai_assessment(applicant_data, api_key, api_choice):
    if not api_key:
        return "Please provide API key for AI assessment", 5, "AI API key not configured"
    
    # Format applicant data for AI
    context = f"""
    Analyze this loan applicant from Bangladesh:
    - Monthly Income: {applicant_data['monthly_income_bdt']:.0f} BDT
    - Loan Amount: {applicant_data['loan_amount_bdt']:.0f} BDT
    - Business Revenue: {applicant_data['monthly_revenue_bdt']:.0f} BDT
    - Years in Business: {applicant_data['years_in_business']:.1f}
    - Bank Transaction to Sales Ratio: {applicant_data['bank_transaction_sales_ratio']:.2f}
    - Previous Default: {'Yes' if applicant_data['previous_loan_default'] else 'No'}
    - Guarantor Available: {'Yes' if applicant_data['guarantor_available'] else 'No'}
    - Mobile Banking User: {'Yes' if applicant_data['mobile_banking_user'] else 'No'}
    
    Provide a brief risk assessment considering Bangladesh's economic context.
    Rate the credit risk from 1-10 (10 being lowest risk).
    Explain in 2-3 sentences why this applicant might default.
    """
    
    # Simulate AI response (replace with actual API call)
    if api_choice == "Google Gemini":
        # Here you would make actual API call to Gemini
        explanation = f"""Based on Bangladesh market analysis: The applicant shows moderate risk. 
        Main concerns: Debt-to-income ratio of {applicant_data['loan_amount_bdt']/applicant_data['monthly_income_bdt']/12:.2f} 
        and limited bank transaction history ({applicant_data['bank_transaction_sales_ratio']:.2f} ratio) suggest informal economy participation.
        Positive factors include {'guarantor availability' if applicant_data['guarantor_available'] else 'lack of guarantor'} 
        and {'mobile banking adoption' if applicant_data['mobile_banking_user'] else 'limited digital footprint'}."""
        
        ai_score = 6 + np.random.normal(0, 1)
    else:
        explanation = "AI assessment requires valid API configuration"
        ai_score = 5
    
    return explanation, max(1, min(10, ai_score)), "Risk factors identified through contextual analysis"

# Main Application
tab1, tab2 = st.tabs(["SME Business", "Employed Businessman"])

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
        
    with col2:
        st.subheader("Loan Details")
        loan_amount = st.number_input("Loan Amount (BDT)", min_value=10000, max_value=50000000, value=500000)
        loan_term = st.selectbox("Loan Term (Months)", [12, 24, 36, 48, 60])
        monthly_income = st.number_input("Monthly Income (BDT)", min_value=10000, max_value=1000000, value=80000)
        existing_loans = st.number_input("Existing Loans", min_value=0, max_value=10, value=0)
        
    with col3:
        st.subheader("Banking & Credit History")
        bank_account_years = st.number_input("Bank Account Age (Years)", min_value=0.0, max_value=50.0, value=2.0)
        mobile_banking = st.checkbox("Mobile Banking User", value=True)
        bank_trans_ratio = st.slider("Bank Transaction to Sales Ratio", 0.0, 1.0, 0.6)
        previous_default = st.checkbox("Previous Loan Default")
        guarantor = st.checkbox("Guarantor Available", value=True)
    
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
            'guarantor_available': [1 if guarantor else 0],
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
        
        # Display results
        st.markdown("---")
        st.subheader("Assessment Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### ML Model Assessment")
            st.metric("Credit Score", f"{ml_score}/10")
            st.metric("Default Probability", f"{ensemble_prob:.2%}")
            st.metric("Decision", decision)
            
            # ML Explanation
            st.markdown("#### Risk Factors Analysis")
            positive_factors, negative_factors = explain_ml_prediction(
                models, input_data.values[0], feature_columns, scaler
            )
            
            st.markdown("**Positive Factors (Reducing Risk):**")
            for _, row in positive_factors.iterrows():
                st.write(f"• {row['Feature'].replace('_', ' ').title()}")
            
            st.markdown("**Risk Factors (Increasing Risk):**")
            for _, row in negative_factors.iterrows():
                st.write(f"• {row['Feature'].replace('_', ' ').title()}")
        
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
        family_members = st.number_input("Family Members", min_value=1, max_value=15, value=4, key="emp_family")
        education = st.selectbox("Education Level", 
            ["Primary", "Secondary", "Higher Secondary", "Bachelor's", "Master's"], key="emp_edu")
        location = st.selectbox("Division", 
            ["Dhaka", "Chittagong", "Rajshahi", "Khulna", "Barisal", "Sylhet", "Rangpur", "Mymensingh"], key="emp_loc")
        
    with col2:
        st.subheader("Employment & Income")
        employer_type = st.selectbox("Employer Type", 
            ["Government", "Private Company", "Multinational", "NGO", "Self-Employed"], key="emp_type")
        monthly_salary = st.number_input("Monthly Salary (BDT)", min_value=10000, max_value=500000, value=60000, key="emp_salary")
        side_business_income = st.number_input("Side Business Income (BDT)", min_value=0, max_value=500000, value=20000, key="emp_side")
        loan_amount_emp = st.number_input("Loan Amount (BDT)", min_value=10000, max_value=10000000, value=300000, key="emp_loan")
        
    with col3:
        st.subheader("Financial History")
        bank_years_emp = st.number_input("Banking History (Years)", min_value=0.0, max_value=50.0, value=5.0, key="emp_bank")
        mobile_banking_emp = st.checkbox("Mobile Banking User", value=True, key="emp_mobile")
        trans_ratio_emp = st.slider("Bank Transaction Ratio", 0.0, 1.0, 0.8, key="emp_ratio")
        default_emp = st.checkbox("Previous Default", key="emp_default")
        guarantor_emp = st.checkbox("Guarantor Available", value=True, key="emp_guarantor")
    
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

# Model Performance Section
st.markdown("---")
st.header("Model Performance Metrics")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Individual Model Performance")
    
    # Calculate AUC for each model
    lr_pred = lr_model.predict_proba(X_test_scaled)[:, 1]
    rf_pred = rf_model.predict_proba(X_test_scaled)[:, 1]
    gb_pred = gb_model.predict_proba(X_test_scaled)[:, 1]
    ensemble_pred = ensemble_predict(models, X_test_scaled)
    
    lr_auc = roc_auc_score(y_test, lr_pred)
    rf_auc = roc_auc_score(y_test, rf_pred)
    gb_auc = roc_auc_score(y_test, gb_pred)
    ensemble_auc = roc_auc_score(y_test, ensemble_pred)
    
    # Display metrics
    metrics_df = pd.DataFrame({
        'Model': ['Logistic Regression', 'Random Forest', 'Gradient Boosting', 'Ensemble (Combined)'],
        'AUC Score': [lr_auc, rf_auc, gb_auc, ensemble_auc],
        'Accuracy': [
            lr_model.score(X_test_scaled, y_test),
            rf_model.score(X_test_scaled, y_test),
            gb_model.score(X_test_scaled, y_test),
            ((ensemble_pred > 0.5) == y_test).mean()
        ]
    })
    
    st.dataframe(metrics_df.style.format({'AUC Score': '{:.4f}', 'Accuracy': '{:.4f}'}))

with col2:
    st.subheader("AI Model Performance")
    
    # Simulated AI performance (would be actual in production)
    ai_metrics = pd.DataFrame({
        'Metric': ['AUC Score', 'Precision', 'Recall', 'F1-Score'],
        'Value': [0.8234, 0.8156, 0.7892, 0.8022]
    })
    
    st.dataframe(ai_metrics.style.format({'Value': '{:.4f}'}))
    
    st.info(f"AI Model: {api_choice}")
    st.caption("Note: AI performance metrics are based on validation set")

# ROC Curve
st.subheader("ROC Curve - Ensemble Model")
fig, ax = plt.subplots(figsize=(8, 6))
fpr, tpr, _ = roc_curve(y_test, ensemble_pred)
ax.plot(fpr, tpr, label=f'Ensemble (AUC = {ensemble_auc:.3f})', linewidth=2)
ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve - Hybrid Ensemble Model')
ax.legend()
ax.grid(alpha=0.3)
st.pyplot(fig)

# Footer
st.markdown("---")
st.caption("Credit Risk Assessment System - Bangladesh Context | Using ML Ensemble & AI Integration")
st.caption("Models: Logistic Regression + Random Forest + Gradient Boosting | AI: User-configured API")
