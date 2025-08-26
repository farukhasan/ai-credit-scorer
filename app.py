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
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="CreditIQ - Bangladesh Credit Scorer",
    layout="wide",
    initial_sidebar_state="collapsed",
    page_icon="💳"
)

# Enhanced Custom CSS for consistent UI
st.markdown("""
    <style>
    /* Main container styling */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
    }
    
    .main > div {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        margin: 20px;
        padding: 30px;
        box-shadow: 0 20px 40px rgba(0,0,0,0.1);
    }
    
    /* Typography */
    h1 {
        background: linear-gradient(45deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        font-size: 3rem !important;
        margin-bottom: 0.5rem !important;
        font-weight: 800 !important;
    }
    
    h2, h3 {
        color: #2c3e50;
        font-weight: 600;
        margin-top: 2rem !important;
    }
    
    .subtitle {
        text-align: center;
        color: #7f8c8d;
        font-size: 1.2rem;
        margin-bottom: 2rem;
        font-weight: 500;
    }
    
    /* Card styling */
    .info-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        margin-bottom: 1.5rem;
        border: 1px solid #e8ecf0;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .info-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.12);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea22, #764ba222);
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
        border: 2px solid transparent;
        background-clip: padding-box;
    }
    
    .result-card {
        background: white;
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        border: 3px solid;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 12px 30px;
        font-weight: 600;
        font-size: 16px;
        width: 100%;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
        background: linear-gradient(45deg, #5a67d8, #6b46c1);
    }
    
    /* Input styling */
    .stSelectbox > div > div {
        border-radius: 10px;
        border: 2px solid #e8ecf0;
        transition: border-color 0.2s ease;
    }
    
    .stSelectbox > div > div:focus-within {
        border-color: #667eea;
        box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.1);
    }
    
    .stNumberInput > div > div > input {
        border-radius: 10px;
        border: 2px solid #e8ecf0;
        transition: border-color 0.2s ease;
    }
    
    .stNumberInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.1);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: transparent;
        padding: 0;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: white;
        border-radius: 15px;
        padding: 12px 24px;
        font-weight: 600;
        border: 2px solid #e8ecf0;
        transition: all 0.2s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white !important;
        border-color: transparent;
    }
    
    /* Progress bar styling */
    .progress-bar {
        background: #e8ecf0;
        height: 8px;
        border-radius: 4px;
        overflow: hidden;
        margin: 8px 0;
    }
    
    .progress-fill {
        height: 100%;
        background: linear-gradient(45deg, #667eea, #764ba2);
        border-radius: 4px;
        transition: width 0.3s ease;
    }
    
    /* Status indicators */
    .status-approve { 
        border-color: #27ae60; 
        background: linear-gradient(135deg, #27ae6022, #2ecc7122);
    }
    .status-review { 
        border-color: #f39c12; 
        background: linear-gradient(135deg, #f39c1222, #e67e2222);
    }
    .status-decline { 
        border-color: #e74c3c; 
        background: linear-gradient(135deg, #e74c3c22, #c0392b22);
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main > div {
            margin: 10px;
            padding: 20px;
        }
        
        h1 {
            font-size: 2rem !important;
        }
    }
    
    /* Remove default streamlit styling */
    .css-1d391kg {
        padding: 0;
    }
    
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Loading animation */
    .loading-spinner {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 3px solid rgba(102, 126, 234, 0.3);
        border-radius: 50%;
        border-top-color: #667eea;
        animation: spin 1s ease-in-out infinite;
    }
    
    @keyframes spin {
        to { transform: rotate(360deg); }
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1>CreditIQ</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Advanced Credit Risk Intelligence System for Bangladesh</p>', unsafe_allow_html=True)
st.markdown("---")

# Initialize session state
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False

# Simulated API key (in real implementation, use st.secrets)
api_key = "demo_key"  # Replace with actual API integration
api_choice = "OpenAI GPT"

# Generate synthetic Bangladesh context data
@st.cache_data
def generate_bangladesh_data(n_samples=1500):
    """Generate synthetic credit data for Bangladesh context"""
    np.random.seed(42)
    
    data = pd.DataFrame({
        # Demographic features
        'age': np.random.randint(22, 65, n_samples),
        'monthly_income_bdt': np.random.exponential(35000, n_samples) + 15000,
        'family_members': np.random.poisson(4, n_samples) + 1,
        'education_level': np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.2, 0.25, 0.3, 0.15, 0.1]),
        
        # Location
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
        
        # Social features
        'guarantor_available': np.random.choice([0, 1], n_samples, p=[0.4, 0.6]),
        'social_capital_score': np.random.beta(5, 2, n_samples),
        'religious_donations_regular': np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
    })
    
    # Create target variable
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

def create_features(df):
    """Create engineered features"""
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

# Define features
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

# Split and scale data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

@st.cache_resource
def train_models():
    """Train ensemble of ML models"""
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train_scaled, y_train)
    
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    rf_model.fit(X_train_scaled, y_train)
    
    gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42, max_depth=5)
    gb_model.fit(X_train_scaled, y_train)
    
    return lr_model, rf_model, gb_model

def ensemble_predict(models, X, weights=[0.3, 0.35, 0.35]):
    """Make ensemble predictions"""
    predictions = []
    for model, weight in zip(models, weights):
        pred_proba = model.predict_proba(X)[:, 1]
        predictions.append(pred_proba * weight)
    
    ensemble_pred = np.sum(predictions, axis=0)
    return ensemble_pred

def calculate_credit_score(default_prob):
    """Convert default probability to credit score (1-10)"""
    score = 10 - (default_prob * 10)
    return max(1, min(10, round(score, 1)))

def get_decision(score):
    """Get loan decision based on score"""
    if score >= 7:
        return "APPROVE", "#27ae60", "status-approve"
    elif score >= 4:
        return "REVIEW REQUIRED", "#f39c12", "status-review"
    else:
        return "DECLINE", "#e74c3c", "status-decline"

def display_result_card(decision, score, color, css_class, loan_amount=None, recommended_amount=None):
    """Display standardized result card"""
    recommendation_text = ""
    if loan_amount and recommended_amount and recommended_amount != loan_amount:
        recommendation_text = f"<div style='margin-top: 15px; font-size: 14px; color: #7f8c8d;'>Recommended Amount: ৳{recommended_amount:,.0f}</div>"
    
    st.markdown(f"""
    <div class='result-card {css_class}' style='border-color: {color};'>
        <h2 style='color: {color}; margin: 0; font-size: 2rem;'>{decision}</h2>
        <div style='font-size: 1.5rem; margin: 15px 0; color: #2c3e50;'>
            Credit Score: <strong>{score}/10</strong>
        </div>
        {recommendation_text}
    </div>
    """, unsafe_allow_html=True)

def display_score_breakdown(ml_score, ai_score=None, ensemble_prob=None):
    """Display score breakdown in cards"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class='metric-card'>
            <h4 style='margin: 0; color: #2c3e50;'>ML Score</h4>
            <div style='font-size: 2rem; font-weight: bold; color: #667eea;'>{ml_score}/10</div>
            <div style='font-size: 0.9rem; color: #7f8c8d;'>Machine Learning</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        if ai_score is not None:
            st.markdown(f"""
            <div class='metric-card'>
                <h4 style='margin: 0; color: #2c3e50;'>AI Score</h4>
                <div style='font-size: 2rem; font-weight: bold; color: #764ba2;'>{ai_score:.1f}/10</div>
                <div style='font-size: 0.9rem; color: #7f8c8d;'>AI Assessment</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='metric-card'>
                <h4 style='margin: 0; color: #2c3e50;'>Default Risk</h4>
                <div style='font-size: 2rem; font-weight: bold; color: #e74c3c;'>{ensemble_prob:.1%}</div>
                <div style='font-size: 0.9rem; color: #7f8c8d;'>Probability</div>
            </div>
            """, unsafe_allow_html=True)
    
    with col3:
        combined_score = (ml_score + (ai_score or 0)) / (2 if ai_score else 1)
        st.markdown(f"""
        <div class='metric-card'>
            <h4 style='margin: 0; color: #2c3e50;'>Combined</h4>
            <div style='font-size: 2rem; font-weight: bold; color: #27ae60;'>{combined_score:.1f}/10</div>
            <div style='font-size: 0.9rem; color: #7f8c8d;'>Final Score</div>
        </div>
        """, unsafe_allow_html=True)

def create_input_form(form_type="sme"):
    """Create standardized input form"""
    if form_type == "sme":
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### Business Information")
            business_type = st.selectbox("Business Type", 
                ["Retail Trade", "Manufacturing", "Services", "Agriculture", "Technology"], key=f"{form_type}_biz")
            years_in_business = st.number_input("Years in Business", min_value=0.0, max_value=50.0, value=3.0, key=f"{form_type}_years")
            monthly_revenue = st.number_input("Monthly Revenue (BDT)", min_value=10000, max_value=10000000, value=150000, key=f"{form_type}_revenue")
            employees = st.number_input("Number of Employees", min_value=0, max_value=500, value=5, key=f"{form_type}_emp")
            
        with col2:
            st.markdown("#### Financial Details")
            loan_amount = st.number_input("Loan Amount (BDT)", min_value=10000, max_value=50000000, value=500000, key=f"{form_type}_loan")
            loan_term = st.selectbox("Loan Term (Months)", [12, 24, 36, 48, 60], key=f"{form_type}_term")
            existing_loans = st.number_input("Existing Loans", min_value=0, max_value=10, value=0, key=f"{form_type}_existing")
            location_type = st.selectbox("Location Type", ["Urban", "Semi-Urban", "Rural"], key=f"{form_type}_location")
            
        with col3:
            st.markdown("#### Credit & Banking")
            bank_account_years = st.number_input("Bank Account Age (Years)", min_value=0.0, max_value=50.0, value=2.0, key=f"{form_type}_bank")
            mobile_banking = st.checkbox("Mobile Banking User", value=True, key=f"{form_type}_mobile")
            previous_default = st.checkbox("Previous Loan Default", key=f"{form_type}_default")
            guarantor = st.selectbox("Guarantor Available", ["No", "Yes"], index=1, key=f"{form_type}_guarantor")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        return {
            'business_type': business_type,
            'years_in_business': years_in_business,
            'monthly_revenue': monthly_revenue,
            'employees': employees,
            'loan_amount': loan_amount,
            'loan_term': loan_term,
            'existing_loans': existing_loans,
            'location_type': location_type,
            'bank_account_years': bank_account_years,
            'mobile_banking': mobile_banking,
            'previous_default': previous_default,
            'guarantor': guarantor
        }
    
    else:  # employed form
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### Personal Information")
            age = st.number_input("Age", min_value=18, max_value=70, value=35, key=f"{form_type}_age")
            education = st.selectbox("Education Level", 
                ["Primary", "Secondary", "Higher Secondary", "Bachelor's", "Master's"], key=f"{form_type}_edu")
            location = st.selectbox("Division", 
                ["Dhaka", "Chittagong", "Rajshahi", "Khulna", "Barisal", "Sylhet", "Rangpur", "Mymensingh"], key=f"{form_type}_div")
            location_type = st.selectbox("Location Type", ["Urban", "Semi-Urban", "Rural"], key=f"{form_type}_loc_type")
            
        with col2:
            st.markdown("#### Employment & Income")
            employer_type = st.selectbox("Employer Type", 
                ["Government", "Private Company", "Multinational", "NGO", "Self-Employed"], key=f"{form_type}_employer")
            monthly_salary = st.number_input("Monthly Salary (BDT)", min_value=10000, max_value=500000, value=60000, key=f"{form_type}_salary")
            side_business_income = st.number_input("Side Business Income (BDT)", min_value=0, max_value=500000, value=20000, key=f"{form_type}_side")
            loan_amount = st.number_input("Loan Amount (BDT)", min_value=10000, max_value=10000000, value=300000, key=f"{form_type}_loan_amt")
            
        with col3:
            st.markdown("#### Credit & Banking")
            loan_term = st.selectbox("Loan Term (Months)", [12, 24, 36, 48, 60], key=f"{form_type}_loan_term")
            bank_years = st.number_input("Banking History (Years)", min_value=0.0, max_value=50.0, value=5.0, key=f"{form_type}_bank_years")
            mobile_banking = st.checkbox("Mobile Banking User", value=True, key=f"{form_type}_mobile_bank")
            previous_default = st.checkbox("Previous Default", key=f"{form_type}_prev_default")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        return {
            'age': age,
            'education': education,
            'location': location,
            'location_type': location_type,
            'employer_type': employer_type,
            'monthly_salary': monthly_salary,
            'side_business_income': side_business_income,
            'loan_amount': loan_amount,
            'loan_term': loan_term,
            'bank_years': bank_years,
            'mobile_banking': mobile_banking,
            'previous_default': previous_default
        }

def get_ai_assessment_simplified(applicant_data, is_sme=True):
    """Simplified AI assessment without external API"""
    # Calculate key financial ratios
    if is_sme:
        income_key = 'monthly_revenue'
        debt_service_ratio = applicant_data['loan_amount'] / (applicant_data['monthly_revenue'] * 12)
        coverage_ratio = (applicant_data['monthly_revenue'] * 12) / applicant_data['loan_amount']
        stability_factor = min(1, applicant_data['years_in_business'] / 5)
    else:
        total_income = applicant_data['monthly_salary'] + applicant_data['side_business_income']
        debt_service_ratio = applicant_data['loan_amount'] / (total_income * 12)
        coverage_ratio = (total_income * 12) / applicant_data['loan_amount']
        stability_factor = 0.8  # Default for employed
    
    # Calculate AI score
    base_score = 10
    
    # Penalize high debt service ratio
    if debt_service_ratio > 0.5:
        base_score -= 3
    elif debt_service_ratio > 0.3:
        base_score -= 1.5
    
    # Reward good coverage
    if coverage_ratio >= 3:
        base_score += 1
    elif coverage_ratio < 1.5:
        base_score -= 2
    
    # Consider stability
    base_score *= stability_factor
    
    # Previous default penalty
    if applicant_data.get('previous_default', False):
        base_score -= 2
    
    # Banking relationship bonus
    banking_years = applicant_data.get('bank_account_years', applicant_data.get('bank_years', 0))
    if banking_years >= 3:
        base_score += 0.5
    
    # Mobile banking bonus
    if applicant_data.get('mobile_banking', False):
        base_score += 0.3
    
    ai_score = max(1, min(10, base_score))
    
    # Generate explanation
    risk_level = "Low" if ai_score >= 7 else "Moderate" if ai_score >= 4 else "High"
    
    explanation = f"""
    AI Risk Assessment: {risk_level} Risk (Score: {ai_score:.1f}/10)
    
    Key Factors:
    • Debt Service Ratio: {debt_service_ratio:.2f} ({'High' if debt_service_ratio > 0.4 else 'Acceptable'})
    • Income Coverage: {coverage_ratio:.1f}x ({'Strong' if coverage_ratio >= 2 else 'Weak'})
    • Banking History: {banking_years:.1f} years
    • Digital Banking: {'Yes' if applicant_data.get('mobile_banking', False) else 'No'}
    
    {'Strong financial profile with manageable risk.' if ai_score >= 7 else 
     'Moderate risk requiring additional review.' if ai_score >= 4 else 
     'High risk factors present, careful consideration needed.'}
    """
    
    return explanation, ai_score

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

# Main tabs
tab1, tab2, tab3 = st.tabs(["SME Business Loan", "Personal Loan", "Model Performance"])

with tab1:
    st.markdown("## SME Business Loan Assessment")
    
    form_data = create_input_form("sme")
    
    if st.button("Assess Credit Risk", key="sme_assess", use_container_width=True):
        with st.spinner("Analyzing credit risk..."):
            # Prepare input data
            input_data = pd.DataFrame({
                'age': [35],
                'monthly_income_bdt': [form_data['monthly_revenue']],
                'family_members': [4],
                'education_level': [3],
                'division': [1],
                'urban_rural': [1 if form_data['location_type'] == "Urban" else 0],
                'loan_amount_bdt': [form_data['loan_amount']],
                'loan_term_months': [form_data['loan_term']],
                'existing_loans': [form_data['existing_loans']],
                'mobile_banking_user': [1 if form_data['mobile_banking'] else 0],
                'bank_account_years': [form_data['bank_account_years']],
                'business_type': [["Retail Trade", "Manufacturing", "Services", "Agriculture", "Technology"].index(form_data['business_type']) + 1],
                'years_in_business': [form_data['years_in_business']],
                'monthly_revenue_bdt': [form_data['monthly_revenue']],
                'employees': [form_data['employees']],
                'monthly_transactions': [20],
                'avg_transaction_size_bdt': [5000],
                'bank_transaction_sales_ratio': [0.75],
                'previous_loan_default': [1 if form_data['previous_default'] else 0],
                'utility_bill_delays': [0],
                'guarantor_available': [1 if form_data['guarantor'] == "Yes" else 0],
                'social_capital_score': [0.7],
                'religious_donations_regular': [1],
                'debt_to_income_ratio': [form_data['loan_amount'] / (form_data['monthly_revenue'] * 12)],
                'revenue_per_employee': [form_data['monthly_revenue'] / (form_data['employees'] + 1)],
                'transaction_frequency_score': [0.67],
                'digital_banking_score': [(1 if form_data['mobile_banking'] else 0) * form_data['bank_account_years']],
                'credit_risk_score': [(1 if form_data['previous_default'] else 0) * 2 / 3]
            })
            
            # Scale input
            input_scaled = scaler.transform(input_data)
            
            # Get predictions
            ensemble_prob = ensemble_predict(models, input_scaled)[0]
            ml_score = calculate_credit_score(ensemble_prob)
            decision, color, css_class = get_decision(ml_score)
            
            # Get AI assessment
            ai_explanation, ai_score = get_ai_assessment_simplified(form_data, is_sme=True)
            
            # Calculate recommended amount
            recommended_amount = min(form_data['monthly_revenue'] * 6, form_data['loan_amount'])
            if ml_score >= 8:
                recommended_amount = min(form_data['loan_amount'] * 1.2, form_data['monthly_revenue'] * 8)
            elif ml_score < 5:
                recommended_amount = form_data['loan_amount'] * 0.7
            
            # Display results
            st.markdown("---")
            display_result_card(decision, ml_score, color, css_class, form_data['loan_amount'], recommended_amount)
            
            # Score breakdown
            st.markdown("### Score Analysis")
            display_score_breakdown(ml_score, ai_score, ensemble_prob)
            
            # Detailed analysis
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<div class="info-card">', unsafe_allow_html=True)
                st.markdown("#### Risk Factors Analysis")
                
                # Key metrics
                debt_ratio = form_data['loan_amount'] / (form_data['monthly_revenue'] * 12)
                revenue_coverage = (form_data['monthly_revenue'] * 12) / form_data['loan_amount']
                
                metrics = [
                    ("Debt-to-Revenue Ratio", f"{debt_ratio:.2f}", "Good" if debt_ratio < 0.5 else "High"),
                    ("Revenue Coverage", f"{revenue_coverage:.1f}x", "Strong" if revenue_coverage > 2 else "Weak"),
                    ("Business Age", f"{form_data['years_in_business']:.1f} years", "Established" if form_data['years_in_business'] >= 3 else "Growing"),
                    ("Banking History", f"{form_data['bank_account_years']:.1f} years", "Good" if form_data['bank_account_years'] >= 2 else "Limited")
                ]
                
                for metric, value, status in metrics:
                    color_indicator = "#27ae60" if status in ["Good", "Strong", "Established"] else "#f39c12" if status in ["Growing", "Limited"] else "#e74c3c"
                    st.markdown(f"""
                    <div style='display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #eee;'>
                        <span><strong>{metric}:</strong> {value}</span>
                        <span style='color: {color_indicator}; font-weight: 600;'>{status}</span>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="info-card">', unsafe_allow_html=True)
                st.markdown("#### AI Assessment Details")
                st.write(ai_explanation)
                st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown("## Personal Loan Assessment")
    
    form_data = create_input_form("employed")
    
    if st.button("Assess Credit Risk", key="personal_assess", use_container_width=True):
        with st.spinner("Analyzing credit risk..."):
            # Calculate total income
            total_income = form_data['monthly_salary'] + form_data['side_business_income']
            
            # Prepare input data
            input_data = pd.DataFrame({
                'age': [form_data['age']],
                'monthly_income_bdt': [total_income],
                'family_members': [4],
                'education_level': [["Primary", "Secondary", "Higher Secondary", "Bachelor's", "Master's"].index(form_data['education']) + 1],
                'division': [["Dhaka", "Chittagong", "Rajshahi", "Khulna", "Barisal", "Sylhet", "Rangpur", "Mymensingh"].index(form_data['location']) + 1],
                'urban_rural': [1 if form_data['location_type'] == "Urban" else 0],
                'loan_amount_bdt': [form_data['loan_amount']],
                'loan_term_months': [form_data['loan_term']],
                'existing_loans': [0],
                'mobile_banking_user': [1 if form_data['mobile_banking'] else 0],
                'bank_account_years': [form_data['bank_years']],
                'business_type': [1],
                'years_in_business': [0],
                'monthly_revenue_bdt': [form_data['side_business_income']],
                'employees': [0],
                'monthly_transactions': [25],
                'avg_transaction_size_bdt': [3000],
                'bank_transaction_sales_ratio': [0.8],
                'previous_loan_default': [1 if form_data['previous_default'] else 0],
                'utility_bill_delays': [0],
                'guarantor_available': [1],  # Default for personal loans
                'social_capital_score': [0.8],
                'religious_donations_regular': [1],
                'debt_to_income_ratio': [form_data['loan_amount'] / (total_income * 12)],
                'revenue_per_employee': [form_data['side_business_income']],
                'transaction_frequency_score': [0.83],
                'digital_banking_score': [(1 if form_data['mobile_banking'] else 0) * form_data['bank_years']],
                'credit_risk_score': [(1 if form_data['previous_default'] else 0) * 2 / 3]
            })
            
            # Scale and predict
            input_scaled = scaler.transform(input_data)
            ensemble_prob = ensemble_predict(models, input_scaled)[0]
            ml_score = calculate_credit_score(ensemble_prob)
            decision, color, css_class = get_decision(ml_score)
            
            # Get AI assessment
            ai_explanation, ai_score = get_ai_assessment_simplified(form_data, is_sme=False)
            
            # Calculate recommended amount
            recommended_amount = min(total_income * 4, form_data['loan_amount'])
            if ml_score >= 8:
                recommended_amount = min(form_data['loan_amount'] * 1.1, total_income * 5)
            elif ml_score < 5:
                recommended_amount = form_data['loan_amount'] * 0.8
            
            # Display results
            st.markdown("---")
            display_result_card(decision, ml_score, color, css_class, form_data['loan_amount'], recommended_amount)
            
            # Score breakdown
            st.markdown("### Score Analysis")
            display_score_breakdown(ml_score, ai_score, ensemble_prob)
            
            # Detailed analysis
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<div class="info-card">', unsafe_allow_html=True)
                st.markdown("#### Financial Profile")
                
                debt_ratio = form_data['loan_amount'] / (total_income * 12)
                income_coverage = (total_income * 12) / form_data['loan_amount']
                
                metrics = [
                    ("Total Monthly Income", f"৳{total_income:,}", ""),
                    ("Debt-to-Income Ratio", f"{debt_ratio:.2f}", "Good" if debt_ratio < 0.4 else "High"),
                    ("Income Coverage", f"{income_coverage:.1f}x", "Strong" if income_coverage > 2.5 else "Moderate"),
                    ("Banking History", f"{form_data['bank_years']:.1f} years", "Established" if form_data['bank_years'] >= 3 else "Growing"),
                    ("Employment Type", form_data['employer_type'], "")
                ]
                
                for metric, value, status in metrics:
                    if status:
                        color_indicator = "#27ae60" if status in ["Good", "Strong", "Established"] else "#f39c12" if status in ["Moderate", "Growing"] else "#e74c3c"
                        st.markdown(f"""
                        <div style='display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #eee;'>
                            <span><strong>{metric}:</strong> {value}</span>
                            <span style='color: {color_indicator}; font-weight: 600;'>{status}</span>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div style='padding: 8px 0; border-bottom: 1px solid #eee;'>
                            <strong>{metric}:</strong> {value}
                        </div>
                        """, unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="info-card">', unsafe_allow_html=True)
                st.markdown("#### AI Assessment Details")
                st.write(ai_explanation)
                st.markdown('</div>', unsafe_allow_html=True)

with tab3:
    st.markdown("## Model Performance & Benchmarks")
    
    # Calculate performance metrics
    lr_pred = lr_model.predict_proba(X_test_scaled)[:, 1]
    rf_pred = rf_model.predict_proba(X_test_scaled)[:, 1]
    gb_pred = gb_model.predict_proba(X_test_scaled)[:, 1]
    ensemble_pred = ensemble_predict(models, X_test_scaled)
    
    lr_auc = roc_auc_score(y_test, lr_pred)
    rf_auc = roc_auc_score(y_test, rf_pred)
    gb_auc = roc_auc_score(y_test, gb_pred)
    ensemble_auc = roc_auc_score(y_test, ensemble_pred)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.markdown("#### Model Performance Metrics")
        
        models_performance = [
            ("Logistic Regression", lr_auc, lr_model.score(X_test_scaled, y_test)),
            ("Random Forest", rf_auc, rf_model.score(X_test_scaled, y_test)),
            ("Gradient Boosting", gb_auc, gb_model.score(X_test_scaled, y_test)),
            ("Ensemble Model", ensemble_auc, ((ensemble_pred > 0.5) == y_test).mean())
        ]
        
        for model_name, auc, accuracy in models_performance:
            st.markdown(f"""
            <div style='padding: 15px; margin: 10px 0; border-radius: 10px; background: linear-gradient(135deg, #667eea22, #764ba222);'>
                <h5 style='margin: 0; color: #2c3e50;'>{model_name}</h5>
                <div style='margin: 10px 0;'>
                    <div style='display: flex; justify-content: space-between;'>
                        <span>AUC Score:</span>
                        <strong>{auc:.4f}</strong>
                    </div>
                    <div style='display: flex; justify-content: space-between;'>
                        <span>Accuracy:</span>
                        <strong>{accuracy:.4f}</strong>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.markdown("#### Feature Importance (Top 10)")
        
        # Get feature importance from Random Forest
        feature_importance = pd.DataFrame({
            'Feature': feature_columns,
            'Importance': rf_model.feature_importances_
        }).sort_values('Importance', ascending=False).head(10)
        
        for _, row in feature_importance.iterrows():
            feature_name = row['Feature'].replace('_', ' ').title()
            importance = row['Importance']
            
            st.markdown(f"""
            <div style='margin: 8px 0;'>
                <div style='display: flex; justify-content: space-between; margin-bottom: 4px;'>
                    <span style='font-size: 14px;'>{feature_name}</span>
                    <span style='font-size: 12px; color: #7f8c8d;'>{importance:.4f}</span>
                </div>
                <div class='progress-bar'>
                    <div class='progress-fill' style='width: {importance*100:.1f}%;'></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Performance summary
    st.markdown('<div class="info-card">', unsafe_allow_html=True)
    st.markdown("#### System Performance Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Overall Accuracy", f"{((ensemble_pred > 0.5) == y_test).mean():.2%}")
    with col2:
        st.metric("AUC Score", f"{ensemble_auc:.4f}")
    with col3:
        st.metric("F1 Score", f"{f1_score(y_test, ensemble_pred > 0.5):.4f}")
    with col4:
        st.metric("Default Rate", f"{y_test.mean():.2%}")
    
    st.markdown("#### Model Interpretation")
    st.write("""
    The ensemble model combines three different algorithms to provide robust credit scoring:
    
    - **Logistic Regression**: Provides interpretable linear relationships
    - **Random Forest**: Captures non-linear patterns and feature interactions
    - **Gradient Boosting**: Focuses on correcting prediction errors
    
    The system achieves strong performance with balanced accuracy and AUC scores, making it suitable 
    for credit risk assessment in the Bangladesh market context.
    """)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #7f8c8d; padding: 20px;'>"
    "CreditIQ - Advanced Credit Risk Intelligence System<br>"
    "Powered by Machine Learning & AI | Designed for Bangladesh Financial Market"
    "</div>", 
    unsafe_allow_html=True
)
