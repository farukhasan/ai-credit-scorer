import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import google.generativeai as genai
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Bangladesh Credit Scoring System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 2rem;
        border-bottom: 3px solid #00a651;
        padding-bottom: 1rem;
    }
    .score-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin: 10px 0;
    }
    .risk-low { background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%) !important; }
    .risk-medium { background: linear-gradient(135deg, #fa709a 0%, #fee140 100%) !important; }
    .risk-high { background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%) !important; }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False
if 'gemini_api_key' not in st.session_state:
    st.session_state.gemini_api_key = ""

def create_synthetic_data(n_samples=1000):
    """Create synthetic credit data with Bangladeshi context variables"""
    np.random.seed(42)
    
    # Bangladeshi specific variables
    districts = ['Dhaka', 'Chittagong', 'Sylhet', 'Rajshahi', 'Khulna', 'Barishal', 'Rangpur', 'Mymensingh']
    banks = ['Dutch Bangla Bank', 'BRAC Bank', 'City Bank', 'Eastern Bank', 'Mutual Trust Bank', 'Prime Bank']
    business_types = ['Textile', 'Garments', 'Agriculture', 'Trading', 'Manufacturing', 'Services', 'IT', 'Food Processing']
    education_levels = ['Primary', 'Secondary', 'HSC', 'Graduate', 'Post Graduate']
    
    data = {
        # Demographics
        'age': np.random.normal(35, 10, n_samples).clip(18, 65),
        'gender': np.random.choice(['Male', 'Female'], n_samples, p=[0.7, 0.3]),
        'marital_status': np.random.choice(['Single', 'Married', 'Divorced'], n_samples, p=[0.3, 0.6, 0.1]),
        'education': np.random.choice(education_levels, n_samples),
        'district': np.random.choice(districts, n_samples),
        'urban_rural': np.random.choice(['Urban', 'Rural'], n_samples, p=[0.4, 0.6]),
        
        # Financial Information
        'monthly_income': np.random.lognormal(10, 0.8, n_samples).clip(15000, 500000),  # BDT
        'existing_loans': np.random.poisson(1.5, n_samples),
        'bank_relationship_years': np.random.gamma(2, 2, n_samples).clip(0, 20),
        'savings_account_balance': np.random.lognormal(8, 1.2, n_samples).clip(0, 1000000),  # BDT
        'credit_history_months': np.random.gamma(3, 8, n_samples).clip(0, 120),
        'previous_defaults': np.random.poisson(0.3, n_samples).clip(0, 5),
        'debt_to_income_ratio': np.random.beta(2, 5, n_samples) * 0.8,
        
        # Business Specific (for SME)
        'business_type': np.random.choice(business_types, n_samples),
        'business_age_years': np.random.gamma(2, 3, n_samples).clip(0, 30),
        'annual_revenue': np.random.lognormal(12, 1, n_samples).clip(100000, 10000000),  # BDT
        'employees_count': np.random.poisson(8, n_samples).clip(1, 100),
        'export_business': np.random.choice(['Yes', 'No'], n_samples, p=[0.3, 0.7]),
        'collateral_value': np.random.lognormal(11, 1.5, n_samples).clip(50000, 5000000),  # BDT
        
        # Employment (for Employed)
        'employment_type': np.random.choice(['Government', 'Private', 'NGO', 'Self-Employed'], n_samples, p=[0.2, 0.4, 0.1, 0.3]),
        'job_stability_years': np.random.gamma(2, 2, n_samples).clip(0, 25),
        'company_size': np.random.choice(['Small', 'Medium', 'Large'], n_samples, p=[0.4, 0.4, 0.2]),
        
        # Loan specific
        'loan_amount': np.random.lognormal(11, 0.8, n_samples).clip(50000, 2000000),  # BDT
        'loan_purpose': np.random.choice(['Business Expansion', 'Working Capital', 'Equipment', 'Home', 'Personal'], n_samples),
        'loan_term_months': np.random.choice([12, 24, 36, 48, 60], n_samples),
        'requested_bank': np.random.choice(banks, n_samples)
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Create target variable (default probability based on risk factors)
    risk_score = (
        (df['age'] < 25) * 0.1 +
        (df['previous_defaults'] > 0) * 0.3 +
        (df['debt_to_income_ratio'] > 0.5) * 0.2 +
        (df['credit_history_months'] < 12) * 0.15 +
        (df['job_stability_years'] < 2) * 0.1 +
        (df['existing_loans'] > 2) * 0.1 +
        (df['urban_rural'] == 'Rural') * 0.05 +
        np.random.normal(0, 0.1, n_samples)
    )
    
    df['default'] = (risk_score > 0.4).astype(int)
    
    return df

def prepare_features(df):
    """Prepare features for ML models"""
    # Select relevant features
    features = [
        'age', 'monthly_income', 'existing_loans', 'bank_relationship_years',
        'savings_account_balance', 'credit_history_months', 'previous_defaults',
        'debt_to_income_ratio', 'business_age_years', 'annual_revenue',
        'employees_count', 'collateral_value', 'job_stability_years',
        'loan_amount', 'loan_term_months'
    ]
    
    # Categorical features to encode
    categorical_features = [
        'gender', 'marital_status', 'education', 'district', 'urban_rural',
        'business_type', 'export_business', 'employment_type', 'company_size',
        'loan_purpose', 'requested_bank'
    ]
    
    # Create a copy for processing
    df_processed = df.copy()
    
    # Encode categorical variables
    label_encoders = {}
    for feature in categorical_features:
        le = LabelEncoder()
        df_processed[feature + '_encoded'] = le.fit_transform(df_processed[feature])
        label_encoders[feature] = le
    
    # Combine features
    all_features = features + [f + '_encoded' for f in categorical_features]
    
    return df_processed[all_features], label_encoders

def train_ensemble_models(X, y):
    """Train ensemble of ML models"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42, max_depth=5)
    }
    
    trained_models = {}
    model_scores = {}
    
    # Train each model
    for name, model in models.items():
        if name == 'Logistic Regression':
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        trained_models[name] = model
        model_scores[name] = accuracy
    
    return trained_models, model_scores, scaler, X_test, y_test

def get_ensemble_prediction(models, scaler, input_data):
    """Get ensemble prediction from all models"""
    predictions = {}
    probabilities = {}
    
    for name, model in models.items():
        if name == 'Logistic Regression':
            input_scaled = scaler.transform([input_data])
            pred = model.predict(input_scaled)[0]
            prob = model.predict_proba(input_scaled)[0][1]
        else:
            pred = model.predict([input_data])[0]
            prob = model.predict_proba([input_data])[0][1]
        
        predictions[name] = pred
        probabilities[name] = prob
    
    # Ensemble prediction (majority vote)
    ensemble_pred = 1 if sum(predictions.values()) >= 2 else 0
    ensemble_prob = np.mean(list(probabilities.values()))
    
    return ensemble_pred, ensemble_prob, predictions, probabilities

def calculate_credit_score(probability, input_data):
    """Calculate credit score out of 10"""
    base_score = (1 - probability) * 10
    
    # Adjust based on other factors
    adjustments = 0
    
    # Positive adjustments
    if input_data[4] > 100000:  # High savings
        adjustments += 0.5
    if input_data[6] == 0:  # No previous defaults
        adjustments += 0.5
    if input_data[3] > 5:  # Long bank relationship
        adjustments += 0.3
    
    # Negative adjustments
    if input_data[7] > 0.6:  # High debt-to-income ratio
        adjustments -= 0.8
    if input_data[2] > 3:  # Many existing loans
        adjustments -= 0.5
    
    final_score = max(1, min(10, base_score + adjustments))
    return round(final_score, 1)

def get_recommendation(score, probability):
    """Get loan recommendation"""
    if score >= 7 and probability < 0.3:
        return "APPROVE", "Low risk applicant with strong financial profile"
    elif score >= 5 and probability < 0.5:
        return "FURTHER REVIEW", "Medium risk applicant - requires additional assessment"
    else:
        return "REJECT", "High risk applicant with significant default probability"

async def get_ai_explanation(gemini_api_key, input_data, feature_names, prediction_prob, applicant_type):
    """Get AI explanation using Gemini API"""
    if not gemini_api_key:
        return "Please provide Gemini API key for AI explanation."
    
    try:
        genai.configure(api_key=gemini_api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Prepare data summary
        data_summary = ""
        important_features = {
            'Age': input_data[0],
            'Monthly Income (BDT)': input_data[1],
            'Existing Loans': input_data[2],
            'Bank Relationship (Years)': input_data[3],
            'Savings Balance (BDT)': input_data[4],
            'Credit History (Months)': input_data[5],
            'Previous Defaults': input_data[6],
            'Debt-to-Income Ratio': input_data[7],
            'Loan Amount (BDT)': input_data[13]
        }
        
        for feature, value in important_features.items():
            data_summary += f"{feature}: {value}\n"
        
        prompt = f"""
        As a credit risk analyst for Bangladesh financial institutions, analyze this {applicant_type} loan application:
        
        Applicant Data:
        {data_summary}
        
        Default Probability: {prediction_prob:.2%}
        
        Please provide:
        1. Key risk factors that contribute to potential default
        2. Positive factors that reduce default risk  
        3. Specific recommendations for risk mitigation in Bangladeshi context
        4. Which variables are most important for this decision
        
        Consider Bangladesh-specific factors like:
        - Rural vs Urban economic conditions
        - Local business environment
        - Banking sector practices
        - Economic stability factors
        
        Keep response concise but insightful (max 300 words).
        """
        
        response = model.generate_content(prompt)
        return response.text
        
    except Exception as e:
        return f"Error getting AI explanation: {str(e)}"

def main():
    st.markdown('<h1 class="main-header">🏦 Bangladesh Credit Scoring System</h1>', unsafe_allow_html=True)
    
    # API Key input
    with st.sidebar:
        st.header("⚙️ Configuration")
        gemini_api_key = st.text_input(
            "Enter Gemini API Key", 
            type="password",
            help="Get free API key from https://ai.google.dev/"
        )
        st.session_state.gemini_api_key = gemini_api_key
        
        if gemini_api_key:
            st.success("✅ Gemini API Connected")
        else:
            st.warning("⚠️ Enter API key for AI explanations")
    
    # Create and load data
    with st.spinner("Loading and preparing data..."):
        if not st.session_state.models_trained:
            # Create synthetic data
            df = create_synthetic_data(2000)
            st.session_state.df = df
            
            # Prepare features and train models
            X, label_encoders = prepare_features(df)
            st.session_state.X = X
            st.session_state.label_encoders = label_encoders
            
            y = df['default']
            models, scores, scaler, X_test, y_test = train_ensemble_models(X, y)
            
            st.session_state.models = models
            st.session_state.model_scores = scores
            st.session_state.scaler = scaler
            st.session_state.X_test = X_test
            st.session_state.y_test = y_test
            st.session_state.models_trained = True
    
    # Tabs for different applicant types
    tab1, tab2, tab3 = st.tabs(["📊 Model Performance", "🏢 SME Business", "👔 Employed Individual"])
    
    with tab1:
        st.header("Model Performance Dashboard")
        
        col1, col2, col3 = st.columns(3)
        
        for i, (model_name, score) in enumerate(st.session_state.model_scores.items()):
            with [col1, col2, col3][i]:
                st.metric(
                    label=f"{model_name} Accuracy",
                    value=f"{score:.3f}",
                    delta=f"{(score-0.8):.3f}" if score > 0.8 else f"{(score-0.8):.3f}"
                )
        
        # Feature importance plot
        if 'Random Forest' in st.session_state.models:
            rf_model = st.session_state.models['Random Forest']
            feature_importance = pd.DataFrame({
                'feature': st.session_state.X.columns,
                'importance': rf_model.feature_importances_
            }).sort_values('importance', ascending=False).head(10)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(data=feature_importance, x='importance', y='feature', ax=ax)
            ax.set_title('Top 10 Feature Importance (Random Forest)')
            st.pyplot(fig)
    
    with tab2:
        st.header("🏢 SME Business Loan Assessment")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Business Information")
            age = st.slider("Age", 18, 65, 35)
            monthly_income = st.number_input("Monthly Income (BDT)", 15000, 500000, 80000)
            business_age = st.slider("Business Age (Years)", 0, 30, 5)
            annual_revenue = st.number_input("Annual Revenue (BDT)", 100000, 10000000, 1200000)
            employees = st.slider("Number of Employees", 1, 100, 8)
            
        with col2:
            st.subheader("Financial Profile")
            existing_loans = st.slider("Existing Loans", 0, 10, 1)
            bank_relationship = st.slider("Bank Relationship (Years)", 0, 20, 3)
            savings_balance = st.number_input("Savings Balance (BDT)", 0, 1000000, 150000)
            credit_history = st.slider("Credit History (Months)", 0, 120, 24)
            previous_defaults = st.slider("Previous Defaults", 0, 5, 0)
            debt_ratio = st.slider("Debt-to-Income Ratio", 0.0, 1.0, 0.3)
            collateral_value = st.number_input("Collateral Value (BDT)", 50000, 5000000, 500000)
            
        col3, col4 = st.columns(2)
        with col3:
            job_stability = st.slider("Job Stability (Years)", 0, 25, 5)
        with col4:
            loan_amount = st.number_input("Requested Loan Amount (BDT)", 50000, 2000000, 300000)
            loan_term = st.selectbox("Loan Term (Months)", [12, 24, 36, 48, 60], index=2)
        
        if st.button("Analyze SME Application", type="primary"):
            # Prepare input data (matching the order of features in training)
            input_data = [
                age, monthly_income, existing_loans, bank_relationship,
                savings_balance, credit_history, previous_defaults, debt_ratio,
                business_age, annual_revenue, employees, collateral_value,
                job_stability, loan_amount, loan_term
            ] + [0] * 11  # Placeholder for encoded categorical features
            
            # Get predictions
            ensemble_pred, ensemble_prob, predictions, probabilities = get_ensemble_prediction(
                st.session_state.models, st.session_state.scaler, input_data
            )
            
            # Calculate credit score
            credit_score = calculate_credit_score(ensemble_prob, input_data)
            recommendation, reason = get_recommendation(credit_score, ensemble_prob)
            
            # Display results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                risk_class = "risk-low" if ensemble_prob < 0.3 else "risk-medium" if ensemble_prob < 0.6 else "risk-high"
                st.markdown(f"""
                <div class="score-box {risk_class}">
                    <h3>Credit Score</h3>
                    <h1>{credit_score}/10</h1>
                    <p>Default Probability: {ensemble_prob:.1%}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                rec_class = "risk-low" if recommendation == "APPROVE" else "risk-medium" if recommendation == "FURTHER REVIEW" else "risk-high"
                st.markdown(f"""
                <div class="score-box {rec_class}">
                    <h3>Recommendation</h3>
                    <h2>{recommendation}</h2>
                    <p>{reason}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="score-box">
                    <h3>Model Predictions</h3>
                    <p>Logistic Regression: {probabilities['Logistic Regression']:.1%}</p>
                    <p>Random Forest: {probabilities['Random Forest']:.1%}</p>
                    <p>Gradient Boosting: {probabilities['Gradient Boosting']:.1%}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # AI Explanation
            st.subheader("🤖 AI Risk Analysis (Powered by Gemini)")
            if st.session_state.gemini_api_key:
                with st.spinner("Getting AI explanation..."):
                    try:
                        ai_explanation = get_ai_explanation(
                            st.session_state.gemini_api_key,
                            input_data,
                            st.session_state.X.columns.tolist(),
                            ensemble_prob,
                            "SME Business"
                        )
                        st.info(ai_explanation)
                    except Exception as e:
                        st.error(f"Error getting AI explanation: {str(e)}")
            else:
                st.warning("Enter Gemini API key in sidebar for AI analysis")
    
    with tab3:
        st.header("👔 Employed Individual Loan Assessment")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Personal Information")
            age_emp = st.slider("Age", 18, 65, 32, key="emp_age")
            monthly_income_emp = st.number_input("Monthly Salary (BDT)", 15000, 200000, 50000, key="emp_income")
            job_stability_emp = st.slider("Job Experience (Years)", 0, 25, 4, key="emp_job_stability")
            
        with col2:
            st.subheader("Financial Profile")
            existing_loans_emp = st.slider("Existing Loans", 0, 10, 0, key="emp_existing_loans")
            bank_relationship_emp = st.slider("Bank Relationship (Years)", 0, 20, 2, key="emp_bank_rel")
            savings_balance_emp = st.number_input("Savings Balance (BDT)", 0, 500000, 80000, key="emp_savings")
            credit_history_emp = st.slider("Credit History (Months)", 0, 120, 18, key="emp_credit_hist")
            previous_defaults_emp = st.slider("Previous Defaults", 0, 5, 0, key="emp_defaults")
            debt_ratio_emp = st.slider("Debt-to-Income Ratio", 0.0, 1.0, 0.25, key="emp_debt_ratio")
        
        col3, col4 = st.columns(2)
        with col3:
            loan_amount_emp = st.number_input("Requested Loan Amount (BDT)", 50000, 1000000, 200000, key="emp_loan_amount")
        with col4:
            loan_term_emp = st.selectbox("Loan Term (Months)", [12, 24, 36, 48, 60], index=1, key="emp_loan_term")
        
        if st.button("Analyze Employee Application", type="primary"):
            # Prepare input data (using default values for business-specific features)
            input_data_emp = [
                age_emp, monthly_income_emp, existing_loans_emp, bank_relationship_emp,
                savings_balance_emp, credit_history_emp, previous_defaults_emp, debt_ratio_emp,
                0, monthly_income_emp * 12, 1, savings_balance_emp,  # Business defaults
                job_stability_emp, loan_amount_emp, loan_term_emp
            ] + [0] * 11  # Placeholder for encoded categorical features
            
            # Get predictions
            ensemble_pred_emp, ensemble_prob_emp, predictions_emp, probabilities_emp = get_ensemble_prediction(
                st.session_state.models, st.session_state.scaler, input_data_emp
            )
            
            # Calculate credit score
            credit_score_emp = calculate_credit_score(ensemble_prob_emp, input_data_emp)
            recommendation_emp, reason_emp = get_recommendation(credit_score_emp, ensemble_prob_emp)
            
            # Display results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                risk_class = "risk-low" if ensemble_prob_emp < 0.3 else "risk-medium" if ensemble_prob_emp < 0.6 else "risk-high"
                st.markdown(f"""
                <div class="score-box {risk_class}">
                    <h3>Credit Score</h3>
                    <h1>{credit_score_emp}/10</h1>
                    <p>Default Probability: {ensemble_prob_emp:.1%}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                rec_class = "risk-low" if recommendation_emp == "APPROVE" else "risk-medium" if recommendation_emp == "FURTHER REVIEW" else "risk-high"
                st.markdown(f"""
                <div class="score-box {rec_class}">
                    <h3>Recommendation</h3>
                    <h2>{recommendation_emp}</h2>
                    <p>{reason_emp}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="score-box">
                    <h3>Model Predictions</h3>
                    <p>Logistic Regression: {probabilities_emp['Logistic Regression']:.1%}</p>
                    <p>Random Forest: {probabilities_emp['Random Forest']:.1%}</p>
                    <p>Gradient Boosting: {probabilities_emp['Gradient Boosting']:.1%}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # AI Explanation
            st.subheader("🤖 AI Risk Analysis (Powered by Gemini)")
            if st.session_state.gemini_api_key:
                with st.spinner("Getting AI explanation..."):
                    try:
                        ai_explanation_emp = get_ai_explanation(
                            st.session_state.gemini_api_key,
                            input_data_emp,
                            st.session_state.X.columns.tolist(),
                            ensemble_prob_emp,
                            "Employed Individual"
                        )
                        st.info(ai_explanation_emp)
                    except Exception as e:
                        st.error(f"Error getting AI explanation: {str(e)}")
            else:
                st.warning("Enter Gemini API key in sidebar for AI analysis")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **Technical Details:**
    - **AI API Used:** Google Gemini 1.5 Flash (Free Tier)
    - **ML Models:** Ensemble of Logistic Regression, Random Forest, and Gradient Boosting
    - **Data Context:** Bangladesh-specific variables including districts, banks, and economic factors
    - **Scoring Range:** 1-10 scale with risk-based recommendations
    
    *This system is for demonstration purposes. Real credit decisions require comprehensive due diligence.*
    """)

if __name__ == "__main__":
    main()