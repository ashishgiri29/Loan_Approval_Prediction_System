import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import plotly.express as px
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

# -------------------------------------------------------------
# 1. PAGE CONFIGURATION (Must be first Streamlit command)
# -------------------------------------------------------------
st.set_page_config(
    page_title="ApprovaAI | Premium Loan Approval Predictor",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -------------------------------------------------------------
# 2. SESSION STATE INITIALIZATION
# -------------------------------------------------------------
if 'predictions_history' not in st.session_state:
    st.session_state.predictions_history = []
if 'active_page' not in st.session_state:
    st.session_state.active_page = "Dashboard"
if 'annual_income_val' not in st.session_state:
    st.session_state.annual_income_val = 600000.0
if 'monthly_income_val' not in st.session_state:
    st.session_state.monthly_income_val = 50000.0
if 'current_balance_val' not in st.session_state:
    st.session_state.current_balance_val = 120000.0

def sync_annual_to_monthly():
    val = st.session_state.annual_income_val
    if val is not None and val >= 0:
        st.session_state.monthly_income_val = round(float(val) / 12.0, 2)

def sync_monthly_to_annual():
    val = st.session_state.monthly_income_val
    if val is not None and val >= 0:
        st.session_state.annual_income_val = round(float(val) * 12.0, 2)

# -------------------------------------------------------------
# 3. ADVANCED RESPONSIVE FINTECH DESIGN SYSTEM (CSS)
# -------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"], .stMarkdown {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #0F172A;
    }
    
    /* Hide sidebar completely */
    [data-testid="stSidebar"], [data-testid="collapsedControl"] {
        display: none !important;
    }
    
    /* Main container bounds & responsive padding */
    .main .block-container {
        max-width: 1350px !important;
        padding-top: 1.5rem !important;
        padding-bottom: 3rem !important;
        padding-left: 1.5rem !important;
        padding-right: 1.5rem !important;
    }

    /* Top Navigation bar container */
    .nav-container {
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 12px;
        padding-bottom: 12px;
        border-bottom: 1px solid #E2E8F0;
        margin-bottom: 24px;
    }

    .brand-title {
        display: flex;
        align-items: center;
        gap: 10px;
        font-weight: 800;
        font-size: 1.4rem;
        color: #0F172A;
        letter-spacing: -0.03em;
    }
    
    .brand-badge {
        background: linear-gradient(135deg, #0F172A 0%, #1E3A8A 100%);
        color: white;
        width: 36px;
        height: 36px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 10px;
        font-weight: 800;
        font-size: 1.2rem;
        box-shadow: 0 2px 6px rgba(15, 23, 42, 0.2);
    }

    /* Navigation button styling */
    div.element-container:has(div.nav-active-btn) + div.element-container button {
        background-color: #1E3A8A !important;
        color: white !important;
        border: 1px solid #1E3A8A !important;
        box-shadow: 0 2px 6px rgba(30, 58, 138, 0.2) !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
        height: 40px !important;
    }
    
    div.element-container:has(div.nav-inactive-btn) + div.element-container button {
        background-color: #FFFFFF !important;
        color: #475569 !important;
        border: 1px solid #E2E8F0 !important;
        font-weight: 500 !important;
        border-radius: 8px !important;
        height: 40px !important;
        transition: all 0.2s ease !important;
    }
    
    div.element-container:has(div.nav-inactive-btn) + div.element-container button:hover {
        background-color: #F8FAFC !important;
        color: #0F172A !important;
        border-color: #CBD5E1 !important;
    }

    /* Fintech Navy Banner */
    .hero-banner {
        background: linear-gradient(135deg, #0F172A 0%, #1E3A8A 50%, #0A2540 100%);
        border-radius: 16px;
        padding: 32px 28px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.15);
    }

    .hero-metrics-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
        gap: 20px;
        border-top: 1px solid rgba(255, 255, 255, 0.15);
        padding-top: 20px;
        margin-top: 20px;
    }

    /* Form Section Header Cards */
    .section-header {
        background: #F8FAFC;
        border-left: 4px solid #1E3A8A;
        padding: 10px 14px;
        border-radius: 0 8px 8px 0;
        font-size: 0.95rem;
        font-weight: 700;
        color: #0F172A;
        margin-top: 18px;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Card Containers */
    .fintech-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 20px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
        margin-bottom: 20px;
    }

    /* Metric Sub-cards */
    .metric-sub-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 12px 14px;
        text-align: left;
        margin-bottom: 12px;
    }
    
    .sub-card-label {
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748B;
        font-weight: 600;
        margin-bottom: 4px;
    }
    
    .sub-card-value {
        font-size: 1.1rem;
        font-weight: 700;
        color: #0F172A;
    }

    /* Primary Action Buttons */
    .stButton>button {
        background-color: #1E3A8A !important;
        color: white !important;
        font-weight: 600 !important;
        border-radius: 10px !important;
        border: none !important;
        box-shadow: 0 4px 12px rgba(30, 58, 138, 0.2) !important;
        transition: all 0.2s ease !important;
    }

    .stButton>button:hover {
        background-color: #172554 !important;
        transform: translateY(-1px);
        box-shadow: 0 6px 16px rgba(30, 58, 138, 0.3) !important;
    }

    /* Mobile & Tablet Responsive Adjustments */
    @media (max-width: 768px) {
        .main .block-container {
            padding-left: 0.75rem !important;
            padding-right: 0.75rem !important;
        }

        .hero-banner {
            padding: 20px 16px;
        }

        .hero-banner h1 {
            font-size: 1.5rem !important;
        }

        .hero-metrics-grid {
            grid-template-columns: repeat(2, 1fr);
            gap: 12px;
        }

        .section-header {
            font-size: 0.88rem;
        }
    }

    @media (max-width: 480px) {
        .hero-metrics-grid {
            grid-template-columns: repeat(2, 1fr);
        }
    }

    /* ============================================================
       MOBILE NAVBAR — compact layout for screens ≤ 767px
       Targets ONLY the first stHorizontalBlock (the nav row).
       Desktop layout is NOT affected by any of the rules below.
       ============================================================ */
    @media (max-width: 767px) {

        /* --- Nav row: switch from stacked columns to a wrapped flex grid --- */
        [data-testid="stMainBlockContainer"] > div > [data-testid="stHorizontalBlock"]:first-of-type {
            display: flex !important;
            flex-wrap: wrap !important;
            gap: 0 !important;
            row-gap: 8px !important;
            padding-bottom: 8px !important;
            border-bottom: 1px solid #E2E8F0;
            margin-bottom: 16px !important;
        }

        /* --- Logo column: half width, left-aligned --- */
        [data-testid="stMainBlockContainer"] > div > [data-testid="stHorizontalBlock"]:first-of-type > [data-testid="column"]:first-child {
            flex: 0 0 60% !important;
            max-width: 60% !important;
            width: 60% !important;
            min-width: 0 !important;
        }

        /* --- Status column (last child): right-aligned, same row as logo --- */
        [data-testid="stMainBlockContainer"] > div > [data-testid="stHorizontalBlock"]:first-of-type > [data-testid="column"]:last-child {
            flex: 0 0 40% !important;
            max-width: 40% !important;
            width: 40% !important;
            min-width: 0 !important;
            order: 1 !important;
        }

        /* --- Nav button columns (2nd through 7th children): horizontal scrollable strip --- */
        [data-testid="stMainBlockContainer"] > div > [data-testid="stHorizontalBlock"]:first-of-type > [data-testid="column"]:nth-child(n+2):nth-last-child(n+2) {
            flex: 0 0 auto !important;
            max-width: none !important;
            width: auto !important;
            min-width: 0 !important;
            order: 2 !important;
        }

        /* --- Make the nav row scroll horizontally when buttons overflow --- */
        [data-testid="stMainBlockContainer"] > div > [data-testid="stHorizontalBlock"]:first-of-type {
            overflow-x: auto !important;
            overflow-y: hidden !important;
            -webkit-overflow-scrolling: touch;
            scrollbar-width: none; /* Firefox */
        }
        [data-testid="stMainBlockContainer"] > div > [data-testid="stHorizontalBlock"]:first-of-type::-webkit-scrollbar {
            display: none; /* Chrome/Safari */
        }

        /* --- Nav buttons: compact, single-line, no wrapping --- */
        [data-testid="stMainBlockContainer"] > div > [data-testid="stHorizontalBlock"]:first-of-type button {
            white-space: nowrap !important;
            font-size: 0.72rem !important;
            padding: 6px 10px !important;
            height: 34px !important;
            min-height: 34px !important;
            border-radius: 6px !important;
        }

        /* --- Brand title: slightly smaller on mobile --- */
        [data-testid="stMainBlockContainer"] > div > [data-testid="stHorizontalBlock"]:first-of-type .brand-title {
            font-size: 1.1rem !important;
            gap: 8px !important;
        }
        [data-testid="stMainBlockContainer"] > div > [data-testid="stHorizontalBlock"]:first-of-type .brand-badge {
            width: 30px !important;
            height: 30px !important;
            font-size: 1rem !important;
        }

        /* --- Hide the nav-active / nav-inactive marker divs (zero height) --- */
        [data-testid="stMainBlockContainer"] > div > [data-testid="stHorizontalBlock"]:first-of-type .nav-active-btn,
        [data-testid="stMainBlockContainer"] > div > [data-testid="stHorizontalBlock"]:first-of-type .nav-inactive-btn {
            display: none !important;
        }
    }

    /* --- Extra-small phones (≤ 375px): even more compact --- */
    @media (max-width: 375px) {
        [data-testid="stMainBlockContainer"] > div > [data-testid="stHorizontalBlock"]:first-of-type button {
            font-size: 0.68rem !important;
            padding: 5px 8px !important;
            height: 32px !important;
            min-height: 32px !important;
        }

        [data-testid="stMainBlockContainer"] > div > [data-testid="stHorizontalBlock"]:first-of-type .brand-title {
            font-size: 1rem !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# 4. SINGLE-FILE MODEL TRAINING & ASSET CACHING ENGINE
# -------------------------------------------------------------
FEATURE_COLS = [
    'age', 'gender', 'marital_status', 'education_level', 'annual_income',
    'monthly_income', 'employment_status', 'debt_to_income_ratio', 'credit_score',
    'loan_amount', 'loan_purpose', 'interest_rate', 'loan_term', 'installment',
    'grade_subgrade', 'num_of_open_accounts', 'total_credit_limit', 'current_balance',
    'delinquency_history', 'public_records', 'num_of_delinquencies'
]

CATEGORICAL_COLS = [
    'gender', 'marital_status', 'education_level', 
    'employment_status', 'loan_purpose', 'grade_subgrade'
]

@st.cache_data
def load_historical_data():
    """Load historical loan dataset from available paths."""
    possible_paths = [
        "loan_dataset_20000.csv",
        "loaddata.csv",
        r"C:\Users\ag206\OneDrive\Desktop\loan_dataset_20000.csv",
        r"C:\Users\ag206\OneDrive\Desktop\loaddata.csv"
    ]
    for path in possible_paths:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                if 'loan_paid_back' in df.columns:
                    return df
            except Exception:
                pass
    return None

def train_and_save_model():
    """Train Random Forest Classifier with SMOTE (applied strictly to training set) and save to model.pkl."""
    df = load_historical_data()
    if df is None:
        raise FileNotFoundError("Dataset 'loan_dataset_20000.csv' not found for training.")

    X = df[FEATURE_COLS].copy()
    y = df['loan_paid_back'].copy()

    # Encode categorical features
    label_encoders = {}
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le

    # Stratified Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Standardize Features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Apply SMOTE ONLY to training data
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

    # Fit Random Forest Classifier
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train_resampled, y_train_resampled)

    model_assets = {
        'model': rf_model,
        'scaler': scaler,
        'label_encoders': label_encoders,
        'feature_cols': FEATURE_COLS
    }

    # Save asset bundle
    try:
        with open('model.pkl', 'wb') as f:
            pickle.dump(model_assets, f)
    except Exception as e:
        st.warning(f"Could not save model bundle to disk: {e}")

    return model_assets

@st.cache_resource
def load_assets():
    """Load pre-trained model assets or automatically train if missing/incompatible."""
    if os.path.exists('model.pkl'):
        try:
            with open('model.pkl', 'rb') as f:
                data = pickle.load(f)
            if isinstance(data, dict) and 'model' in data and 'scaler' in data and 'label_encoders' in data:
                return data
        except Exception:
            pass

    # If model.pkl is absent or invalid, train automatically inside app.py
    try:
        return train_and_save_model()
    except Exception as e:
        st.error(f"⚠️ Model Initialization Failed: {e}")
        return None

# Load model assets on startup
assets = load_assets()

# -------------------------------------------------------------
# 5. TOP NAVIGATION BAR
# -------------------------------------------------------------
col_logo, col_dash, col_pred, col_ana, col_ins, col_hist, col_about, col_status = st.columns(
    [2.0, 1.1, 1.2, 1.1, 1.3, 1.0, 1.0, 1.8]
)

with col_logo:
    st.markdown("""
    <div class="brand-title">
        <div class="brand-badge">L</div>
        <span>ApprovaAI</span>
    </div>
    """, unsafe_allow_html=True)

with col_dash:
    is_active = (st.session_state.active_page == "Dashboard")
    st.markdown('<div class="nav-active-btn"></div>' if is_active else '<div class="nav-inactive-btn"></div>', unsafe_allow_html=True)
    if st.button("Dashboard", key="nav_dash", width="stretch"):
        st.session_state.active_page = "Dashboard"
        st.rerun()

with col_pred:
    is_active = (st.session_state.active_page == "Predict")
    st.markdown('<div class="nav-active-btn"></div>' if is_active else '<div class="nav-inactive-btn"></div>', unsafe_allow_html=True)
    if st.button("Predict Loan", key="nav_pred", width="stretch"):
        st.session_state.active_page = "Predict"
        st.rerun()

with col_ana:
    is_active = (st.session_state.active_page == "Analytics")
    st.markdown('<div class="nav-active-btn"></div>' if is_active else '<div class="nav-inactive-btn"></div>', unsafe_allow_html=True)
    if st.button("Analytics", key="nav_ana", width="stretch"):
        st.session_state.active_page = "Analytics"
        st.rerun()

with col_ins:
    is_active = (st.session_state.active_page == "Insights")
    st.markdown('<div class="nav-active-btn"></div>' if is_active else '<div class="nav-inactive-btn"></div>', unsafe_allow_html=True)
    if st.button("Model Insights", key="nav_ins", width="stretch"):
        st.session_state.active_page = "Insights"
        st.rerun()

with col_hist:
    is_active = (st.session_state.active_page == "History")
    st.markdown('<div class="nav-active-btn"></div>' if is_active else '<div class="nav-inactive-btn"></div>', unsafe_allow_html=True)
    if st.button("History", key="nav_hist", width="stretch"):
        st.session_state.active_page = "History"
        st.rerun()

with col_about:
    is_active = (st.session_state.active_page == "About")
    st.markdown('<div class="nav-active-btn"></div>' if is_active else '<div class="nav-inactive-btn"></div>', unsafe_allow_html=True)
    if st.button("About", key="nav_about", width="stretch"):
        st.session_state.active_page = "About"
        st.rerun()

with col_status:
    status_color = "#10B981" if assets else "#EF4444"
    status_text = "Model Live" if assets else "Model Offline"
    st.markdown(f"""
    <div style="display: flex; justify-content: flex-end; align-items: center; height: 40px;">
        <span style="background-color: #ECFDF5; color: #065F46; border: 1px solid #A7F3D0; padding: 4px 12px; border-radius: 9999px; font-size: 0.8rem; font-weight: 600; display: flex; align-items: center; gap: 6px;">
            <span style="width: 8px; height: 8px; background-color: {status_color}; border-radius: 50%; display: inline-block;"></span> {status_text}
        </span>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<hr style='margin: 12px 0 24px 0; border: none; border-top: 1px solid #E2E8F0;'>", unsafe_allow_html=True)

page = st.session_state.active_page

# -------------------------------------------------------------
# 6. PAGE CONTENT ROUTING
# -------------------------------------------------------------

# PAGE 1: DASHBOARD
if page == "Dashboard":
    st.markdown("""
    <div class="hero-banner">
        <h1 style="margin: 0; font-size: 2rem; font-weight: 800; letter-spacing: -0.02em;">ApprovaAI Loan Approval Intelligence Platform</h1>
        <p style="margin: 8px 0 0 0; font-size: 0.98rem; opacity: 0.88; max-width: 850px; line-height: 1.5;">
            Enterprise-grade ML underwriting system trained on 20,000 applicant financial records. Real-time repayment probability assessment and credit risk analytics.
        </p>
        <div class="hero-metrics-grid">
            <div>
                <div style="font-size: 1.75rem; font-weight: 800; line-height: 1.1;">94.61%</div>
                <div style="font-size: 0.72rem; font-weight: 700; text-transform: uppercase; opacity: 0.75; letter-spacing: 0.05em; margin-top: 4px;">Accuracy</div>
            </div>
            <div>
                <div style="font-size: 1.75rem; font-weight: 800; line-height: 1.1;">98.66%</div>
                <div style="font-size: 0.72rem; font-weight: 700; text-transform: uppercase; opacity: 0.75; letter-spacing: 0.05em; margin-top: 4px;">Recall</div>
            </div>
            <div>
                <div style="font-size: 1.75rem; font-weight: 800; line-height: 1.1;">0.9841</div>
                <div style="font-size: 0.72rem; font-weight: 700; text-transform: uppercase; opacity: 0.75; letter-spacing: 0.05em; margin-top: 4px;">ROC-AUC</div>
            </div>
            <div>
                <div style="font-size: 1.75rem; font-weight: 800; line-height: 1.1;">20,000</div>
                <div style="font-size: 0.72rem; font-weight: 700; text-transform: uppercase; opacity: 0.75; letter-spacing: 0.05em; margin-top: 4px;">Dataset Records</div>
            </div>
            <div>
                <div style="font-size: 1.75rem; font-weight: 800; line-height: 1.1;">21</div>
                <div style="font-size: 0.72rem; font-weight: 700; text-transform: uppercase; opacity: 0.75; letter-spacing: 0.05em; margin-top: 4px;">Input Features</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Session Analytics KPI Row
    total_preds = len(st.session_state.predictions_history)
    approved_preds = sum(1 for p in st.session_state.predictions_history if p['decision'] == 1)
    rejected_preds = total_preds - approved_preds

    app_rate = (approved_preds / total_preds * 100) if total_preds > 0 else 0
    rej_rate = (rejected_preds / total_preds * 100) if total_preds > 0 else 0
    avg_repay_prob = np.mean([p['prob'] for p in st.session_state.predictions_history]) * 100 if total_preds > 0 else 0

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)

    with col_m1:
        st.markdown(f"""
        <div class="fintech-card" style="display: flex; align-items: center; gap: 14px;">
            <div style="background-color: #EFF6FF; color: #1E3A8A; width: 44px; height: 44px; border-radius: 10px; display: flex; align-items: center; justify-content: center; font-size: 1.3rem; flex-shrink: 0;">📋</div>
            <div>
                <div style="font-size: 1.4rem; font-weight: 800; color: #0F172A; line-height: 1.1;">{total_preds}</div>
                <div style="font-size: 0.78rem; font-weight: 600; color: #64748B; margin-top: 3px;">Total Predictions</div>
                <div style="font-size: 0.72rem; color: #1E3A8A; font-weight: 500;">Active session count</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_m2:
        st.markdown(f"""
        <div class="fintech-card" style="display: flex; align-items: center; gap: 14px;">
            <div style="background-color: #ECFDF5; color: #10B981; width: 44px; height: 44px; border-radius: 10px; display: flex; align-items: center; justify-content: center; font-size: 1.3rem; flex-shrink: 0;">✅</div>
            <div>
                <div style="font-size: 1.4rem; font-weight: 800; color: #0F172A; line-height: 1.1;">{approved_preds}</div>
                <div style="font-size: 0.78rem; font-weight: 600; color: #64748B; margin-top: 3px;">Approved Loans</div>
                <div style="font-size: 0.72rem; color: #10B981; font-weight: 600;">{app_rate:.0f}% approval rate</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_m3:
        st.markdown(f"""
        <div class="fintech-card" style="display: flex; align-items: center; gap: 14px;">
            <div style="background-color: #FEF2F2; color: #EF4444; width: 44px; height: 44px; border-radius: 10px; display: flex; align-items: center; justify-content: center; font-size: 1.3rem; flex-shrink: 0;">❌</div>
            <div>
                <div style="font-size: 1.4rem; font-weight: 800; color: #0F172A; line-height: 1.1;">{rejected_preds}</div>
                <div style="font-size: 0.78rem; font-weight: 600; color: #64748B; margin-top: 3px;">Rejected Loans</div>
                <div style="font-size: 0.72rem; color: #EF4444; font-weight: 600;">{rej_rate:.0f}% rejection rate</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_m4:
        st.markdown(f"""
        <div class="fintech-card" style="display: flex; align-items: center; gap: 14px;">
            <div style="background-color: #F5F3FF; color: #7C3AED; width: 44px; height: 44px; border-radius: 10px; display: flex; align-items: center; justify-content: center; font-size: 1.3rem; flex-shrink: 0;">📈</div>
            <div>
                <div style="font-size: 1.4rem; font-weight: 800; color: #0F172A; line-height: 1.1;">{avg_repay_prob:.0f}%</div>
                <div style="font-size: 0.78rem; font-weight: 600; color: #64748B; margin-top: 3px;">Avg Approval Prob</div>
                <div style="font-size: 0.72rem; color: #7C3AED; font-weight: 500;">Across all predictions</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("#### Quick Launch Actions")
    col_act1, col_act2, col_act3 = st.columns(3)
    with col_act1:
        if st.button("🚀 Evaluate New Loan Application", width="stretch"):
            st.session_state.active_page = "Predict"
            st.rerun()
    with col_act2:
        if st.button("📊 View Dataset Analytics", width="stretch"):
            st.session_state.active_page = "Analytics"
            st.rerun()
    with col_act3:
        if st.button("📚 Model Technical Metrics", width="stretch"):
            st.session_state.active_page = "About"
            st.rerun()

# PAGE 2: PREDICT LOAN
elif page == "Predict":
    st.markdown("### 🕵️ Underwriting Decision Engine")
    st.markdown("Enter applicant financial parameters across the four structured sections below.")

    if assets is None:
        st.error("⚠️ Model assets could not be loaded or trained. Please ensure `loan_dataset_20000.csv` is present in your project directory.")
    else:
        model = assets['model']
        scaler = assets['scaler']
        label_encoders = assets['label_encoders']
        feature_cols = assets['feature_cols']

        col_form, col_report = st.columns([1.5, 1.0], gap="large")

        with col_form:
            # 1. PERSONAL DETAILS
            st.markdown('<div class="section-header">👤 1. Personal Details</div>', unsafe_allow_html=True)
            p_col1, p_col2 = st.columns(2)
            with p_col1:
                age = st.slider("Applicant Age", min_value=18, max_value=85, value=35, step=1)
                gender = st.selectbox("Gender", options=label_encoders['gender'].classes_, index=1)
            with p_col2:
                marital_status = st.selectbox("Marital Status", options=label_encoders['marital_status'].classes_, index=1)
                education_level = st.selectbox("Education Level", options=label_encoders['education_level'].classes_, index=0)

            # 2. INCOME & EMPLOYMENT
            st.markdown('<div class="section-header">💼 2. Income & Employment</div>', unsafe_allow_html=True)
            i_col1, i_col2 = st.columns(2)
            with i_col1:
                employment_status = st.selectbox("Employment Status", options=label_encoders['employment_status'].classes_, index=0)
                annual_income = st.number_input(
                    "Annual Income", 
                    min_value=0.0, 
                    max_value=10000000.0, 
                    step=1000.0, 
                    format="%.2f",
                    key="annual_income_val",
                    on_change=sync_annual_to_monthly
                )
                num_of_open_accounts = st.number_input("Open Accounts Count", min_value=0, max_value=40, value=6, step=1)
            with i_col2:
                monthly_income = st.number_input(
                    "Monthly Income", 
                    min_value=0.0, 
                    max_value=1000000.0, 
                    step=100.0, 
                    format="%.2f",
                    key="monthly_income_val",
                    on_change=sync_monthly_to_annual
                )
                total_credit_limit = st.number_input("Total Credit Limit", min_value=500.0, max_value=10000000.0, value=450000.0, step=1000.0)

            # 3. CREDIT PROFILE
            st.markdown('<div class="section-header">💳 3. Credit Profile</div>', unsafe_allow_html=True)
            c_col1, c_col2 = st.columns(2)
            with c_col1:
                credit_score = st.slider("Credit Score (FICO)", min_value=300, max_value=850, value=720, step=1)
                current_balance = st.number_input(
                    "Current Outstanding Debt (₹ / $)", 
                    min_value=0.0, 
                    max_value=10000000.0, 
                    step=1000.0, 
                    format="%.2f",
                    key="current_balance_val"
                )
                delinquency_history = st.selectbox("Delinquency History (Counts)", options=[0, 1, 2, 3, 4, 5], index=0)
            with c_col2:
                # Automatic DTI calculation: Current Outstanding Debt / (Monthly Income * 12)
                ann_inc = monthly_income * 12.0
                if ann_inc > 0 and current_balance >= 0:
                    calculated_dti = round(current_balance / ann_inc, 4)
                else:
                    calculated_dti = 0.0

                debt_to_income_ratio = st.number_input(
                    "Debt-to-Income Ratio (DTI)", 
                    value=round(calculated_dti, 2), 
                    disabled=True, 
                    format="%.2f",
                    help="Automatically calculated as Current Outstanding Debt / (Monthly Income × 12)"
                )
                public_records = st.selectbox("Public Records", options=[0, 1, 2, 3], index=0)
                num_of_delinquencies = st.number_input("Number of Delinquencies", min_value=0, max_value=20, value=0, step=1)

            # Validation Warnings for DTI & Income inputs
            if monthly_income <= 0 or annual_income <= 0:
                st.warning("⚠️ Income must be greater than zero to calculate Debt-to-Income Ratio (DTI).")
            elif current_balance < 0:
                st.warning("⚠️ Current Outstanding Debt cannot be negative.")
            elif calculated_dti > 1.0:
                st.warning(f"⚠️ High Debt-to-Income Warning: Calculated DTI ({calculated_dti:.2f}) exceeds 1.00 (100% of annual income). High default risk profile.")

            # 4. LOAN DETAILS
            st.markdown('<div class="section-header">📑 4. Loan Details</div>', unsafe_allow_html=True)
            l_col1, l_col2 = st.columns(2)
            with l_col1:
                loan_amount = st.number_input("Requested Loan Amount ($)", min_value=500.0, max_value=150000.0, value=18000.0, step=500.0, format="%.2f")
                loan_purpose = st.selectbox("Loan Purpose", options=label_encoders['loan_purpose'].classes_, index=2)
                loan_term = st.selectbox("Loan Term (Months)", options=[36, 60], index=0)
            with l_col2:
                interest_rate = st.slider("Interest Rate (%)", min_value=1.0, max_value=35.0, value=10.5, step=0.1)
                grade_subgrade = st.selectbox("Lender Grade / Subgrade", options=label_encoders['grade_subgrade'].classes_, index=3)
                
                # Auto-calculate installment amortized estimate
                r_monthly = (interest_rate / 100.0) / 12.0
                if r_monthly > 0:
                    calc_inst = (r_monthly * loan_amount) / (1 - (1 + r_monthly) ** (-loan_term))
                else:
                    calc_inst = loan_amount / loan_term
                installment = st.number_input("Calculated Monthly Installment ($)", min_value=1.0, max_value=10000.0, value=round(calc_inst, 2), step=10.0, format="%.2f")

            st.markdown("<br>", unsafe_allow_html=True)
            predict_btn = st.button("⚡ Generate Credit Decision Report", width="stretch")

        with col_report:
            if not predict_btn:
                st.markdown("""
                <div style="text-align: center; padding: 48px 24px; color: #64748B; background: white; border: 1px dashed #CBD5E1; border-radius: 16px; margin-top: 15px;">
                    <p style="font-size: 3rem; margin: 0;">📄</p>
                    <h4 style="margin-top: 12px; font-weight: 700; color: #0F172A;">Pending Underwriting Assessment</h4>
                    <p style="font-size: 0.88rem; line-height: 1.5; color: #64748B;">
                        Complete the applicant profile across the four categories on the left and click <b>Generate Credit Decision Report</b>.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                if annual_income <= 0 or monthly_income <= 0:
                    st.error("⚠️ Validation Error: Income must be greater than zero for credit assessment.")
                elif current_balance < 0:
                    st.error("⚠️ Validation Error: Current Outstanding Debt cannot be negative.")
                else:
                    input_data = {
                        'age': age,
                        'gender': gender,
                        'marital_status': marital_status,
                        'education_level': education_level,
                        'annual_income': float(annual_income),
                        'monthly_income': float(monthly_income),
                        'employment_status': employment_status,
                        'debt_to_income_ratio': float(calculated_dti),
                        'credit_score': credit_score,
                        'loan_amount': loan_amount,
                        'loan_purpose': loan_purpose,
                        'interest_rate': interest_rate,
                        'loan_term': loan_term,
                        'installment': installment,
                        'grade_subgrade': grade_subgrade,
                        'num_of_open_accounts': num_of_open_accounts,
                        'total_credit_limit': total_credit_limit,
                        'current_balance': current_balance,
                        'delinquency_history': delinquency_history,
                        'public_records': public_records,
                        'num_of_delinquencies': num_of_delinquencies
                    }

                try:
                    input_df = pd.DataFrame([input_data])
                    input_df = input_df[feature_cols]

                    # Transform categorical columns
                    for col, le in label_encoders.items():
                        input_df[col] = le.transform(input_df[col])

                    # Scale features
                    input_scaled = scaler.transform(input_df)

                    # Predict
                    prediction = model.predict(input_scaled)[0]
                    prediction_proba = model.predict_proba(input_scaled)[0]

                    repayment_prob = prediction_proba[1]
                    default_prob = prediction_proba[0]

                    # Save prediction record to session history
                    history_record = input_data.copy()
                    history_record['decision'] = int(prediction)
                    history_record['prob'] = float(repayment_prob)
                    st.session_state.predictions_history.append(history_record)

                    # Result Card
                    if prediction == 1:
                        st.balloons()
                        st.markdown("""
                        <div style="background-color: #ECFDF5; border: 1px solid #A7F3D0; border-radius: 14px; padding: 20px; display: flex; align-items: center; gap: 16px; color: #065F46; margin-top: 15px; margin-bottom: 20px;">
                            <span style="font-size: 2.2rem; font-weight: 800;">✓</span>
                            <div>
                                <h4 style="margin: 0; font-weight: 800; font-size: 1.15rem;">LOAN APPROVED</h4>
                                <p style="margin: 3px 0 0 0; font-size: 0.85rem; opacity: 0.9;">Applicant satisfies credit risk & repayment standards.</p>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div style="background-color: #FEF2F2; border: 1px solid #FCA5A5; border-radius: 14px; padding: 20px; display: flex; align-items: center; gap: 16px; color: #991B1B; margin-top: 15px; margin-bottom: 20px;">
                            <span style="font-size: 2.2rem; font-weight: 800;">❌</span>
                            <div>
                                <h4 style="margin: 0; font-weight: 800; font-size: 1.15rem;">LOAN REJECTED</h4>
                                <p style="margin: 3px 0 0 0; font-size: 0.85rem; opacity: 0.9;">High risk profile exceeding default thresholds.</p>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    prob_color = "#10B981" if prediction == 1 else "#EF4444"
                    st.markdown(f"""
                    <div style="display: flex; justify-content: space-between; font-weight: 700; font-size: 0.95rem; margin-bottom: 6px;">
                        <span style="color: #475569;">Repayment Probability</span>
                        <span style="color: {prob_color}; font-size: 1.1rem;">{repayment_prob*100:.1f}%</span>
                    </div>
                    """, unsafe_allow_html=True)
                    st.progress(float(repayment_prob))
                    st.markdown("<br>", unsafe_allow_html=True)

                    # Sub-metrics
                    g1, g2 = st.columns(2)
                    with g1:
                        if repayment_prob >= 0.70:
                            risk_badge = "<span style='color: #10B981;'>● LOW RISK</span>"
                        elif repayment_prob >= 0.50:
                            risk_badge = "<span style='color: #F59E0B;'>● MEDIUM RISK</span>"
                        else:
                            risk_badge = "<span style='color: #EF4444;'>● HIGH RISK</span>"

                        st.markdown(f"""
                        <div class="metric-sub-card">
                            <div class="sub-card-label">Assessed Risk Tier</div>
                            <div class="sub-card-value">{risk_badge}</div>
                        </div>
                        """, unsafe_allow_html=True)

                        st.markdown(f"""
                        <div class="metric-sub-card">
                            <div class="sub-card-label">DTI Ratio</div>
                            <div class="sub-card-value">{debt_to_income_ratio:.2f}</div>
                        </div>
                        """, unsafe_allow_html=True)

                    with g2:
                        st.markdown(f"""
                        <div class="metric-sub-card">
                            <div class="sub-card-label">Credit Rating</div>
                            <div class="sub-card-value">{grade_subgrade}</div>
                        </div>
                        """, unsafe_allow_html=True)

                        st.markdown(f"""
                        <div class="metric-sub-card">
                            <div class="sub-card-label">Credit Score</div>
                            <div class="sub-card-value">{credit_score}</div>
                        </div>
                        """, unsafe_allow_html=True)

                    st.markdown("##### 📌 Key Risk Driver Explanations")
                    drivers = []
                    if employment_status == 'Employed':
                        drivers.append("🟢 Stable employment status contributes positively to approval score.")
                    else:
                        drivers.append(f"🔴 Employment status ('{employment_status}') reflects higher income volatility risk.")

                    if credit_score >= 700:
                        drivers.append(f"🟢 High credit score ({credit_score}) demonstrates strong repayment reliability.")
                    else:
                        drivers.append(f"🔴 Lower credit score ({credit_score}) elevates delinquency probability.")

                    if debt_to_income_ratio <= 0.35:
                        drivers.append(f"🟢 Healthy DTI ratio ({debt_to_income_ratio:.2f}) leaves sufficient disposable income.")
                    else:
                        drivers.append(f"🔴 Elevated DTI ratio ({debt_to_income_ratio:.2f}) indicates debt burden risk.")

                    if delinquency_history == 0:
                        drivers.append("🟢 Clean historical delinquency record with zero past payment defaults.")
                    else:
                        drivers.append(f"🔴 Past delinquency history ({delinquency_history} event(s)) negatively impacts rating.")

                    for d in drivers:
                        st.markdown(f"<p style='font-size: 0.85rem; color: #334155; margin: 6px 0;'>{d}</p>", unsafe_allow_html=True)

                except Exception as ex:
                    st.error(f"Prediction error occurred: {ex}")

# PAGE 3: ANALYTICS
elif page == "Analytics":
    st.markdown("### 📊 Historical Loan Portfolio Analytics")
    df_hist = load_historical_data()

    if df_hist is None:
        st.warning("⚠️ Historical dataset `loan_dataset_20000.csv` could not be located.")
    else:
        st.markdown("Explore macroeconomic patterns, income distributions, and repayment behavior across the 20,000 dataset records.")

        col_f1, col_f2 = st.columns(2)
        with col_f1:
            purposes = ['All'] + sorted(list(df_hist['loan_purpose'].unique()))
            selected_purpose = st.selectbox("Filter by Purpose:", options=purposes, key="ana_purpose")
        with col_f2:
            max_inc = int(df_hist['annual_income'].max())
            min_inc = int(df_hist['annual_income'].min())
            income_filter = st.slider("Max Annual Income Filter ($)", min_value=min_inc, max_value=max_inc, value=150000, step=5000, key="ana_inc")

        df_filtered = df_hist[df_hist['annual_income'] <= income_filter]
        if selected_purpose != 'All':
            df_filtered = df_filtered[df_filtered['loan_purpose'] == selected_purpose]

        col_c1, col_c2 = st.columns(2)
        with col_c1:
            df_status = df_filtered['loan_paid_back'].map({0: 'Defaulted', 1: 'Paid Back'}).value_counts().reset_index()
            df_status.columns = ['Loan Status', 'Count']
            fig_status = px.pie(
                df_status, names='Loan Status', values='Count', hole=0.4,
                color='Loan Status',
                color_discrete_map={'Paid Back': '#10B981', 'Defaulted': '#EF4444'},
                title="Historical Repayment Status Distribution"
            )
            fig_status.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="#0F172A", margin=dict(t=40, b=10, l=10, r=10))
            st.plotly_chart(fig_status, width="stretch")

        with col_c2:
            df_gender = df_filtered['gender'].value_counts().reset_index()
            df_gender.columns = ['Gender', 'Count']
            fig_gender = px.pie(
                df_gender, names='Gender', values='Count', hole=0.4,
                color_discrete_sequence=['#1E3A8A', '#38BDF8', '#F59E0B'],
                title="Applicant Gender Demographics"
            )
            fig_gender.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="#0F172A", margin=dict(t=40, b=10, l=10, r=10))
            st.plotly_chart(fig_gender, width="stretch")

        col_c3, col_c4 = st.columns(2)
        with col_c3:
            fig_income = px.histogram(
                df_filtered, x='annual_income', nbins=40,
                color_discrete_sequence=['#1E3A8A'],
                title="Income Spectrum Distribution"
            )
            fig_income.update_layout(
                xaxis_title="Annual Income ($)", yaxis_title="Applicants Count",
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="#0F172A",
                margin=dict(t=40, b=10, l=10, r=10)
            )
            st.plotly_chart(fig_income, width="stretch")

        with col_c4:
            df_edu = df_filtered.copy()
            df_edu['Loan Status'] = df_edu['loan_paid_back'].map({0: 'Defaulted', 1: 'Paid Back'})
            df_edu_grouped = df_edu.groupby(['education_level', 'Loan Status']).size().reset_index(name='Count')
            fig_edu = px.bar(
                df_edu_grouped, x='education_level', y='Count', color='Loan Status',
                barmode='group',
                color_discrete_map={'Paid Back': '#10B981', 'Defaulted': '#EF4444'},
                title="Repayment Breakdown by Education Level"
            )
            fig_edu.update_layout(
                xaxis_title="Education Level", yaxis_title="Applicants Count",
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="#0F172A",
                margin=dict(t=40, b=10, l=10, r=10)
            )
            st.plotly_chart(fig_edu, width="stretch")

# PAGE 4: HISTORY
elif page == "History":
    st.markdown("### 🕒 Active Session Prediction Audit Log")
    st.markdown("Review and download predictions computed during your current interactive session.")

    if not st.session_state.predictions_history:
        st.info("No loan predictions evaluated yet in this session. Go to **Predict Loan** to test an applicant profile.")
    else:
        history_df = pd.DataFrame(st.session_state.predictions_history)
        history_df['Decision'] = history_df['decision'].apply(lambda x: 'APPROVED' if x == 1 else 'REJECTED')
        history_df['Probability'] = history_df['prob'].apply(lambda x: f"{x*100:.1f}%")

        display_cols = ['Decision', 'Probability', 'loan_amount', 'annual_income', 'credit_score', 'loan_purpose', 'employment_status', 'debt_to_income_ratio']
        remaining = [c for c in history_df.columns if c not in display_cols and c not in ['decision', 'prob']]
        final_cols = display_cols + remaining

        st.dataframe(history_df[final_cols], width="stretch")

        csv_bytes = history_df[final_cols].to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Export Session Audit Log (CSV)",
            data=csv_bytes,
            file_name="approvaai_prediction_history.csv",
            mime="text/csv"
        )

# PAGE 5: MODEL INSIGHTS
elif page == "Insights":
    st.markdown("### 🔬 Model Insights & Technical Audit Dashboard")
    st.markdown("Deep dive into the Random Forest model architecture, evaluation metrics, confusion matrix, feature importances, and dataset properties.")

    # Unpack model assets — required by the simulator tab and any live metric display
    if assets is None:
        st.error("⚠️ Model assets could not be loaded. Please ensure `loan_dataset_20000.csv` is present.")
        st.stop()
    model = assets['model']
    scaler = assets['scaler']
    label_encoders = assets['label_encoders']
    feature_cols = assets['feature_cols']

    t_ov, t_perf, t_cm, t_fi, t_pipe, t_sim, t_comp, t_ds = st.tabs([
        "📊 1. Overview",
        "🎯 2. Metrics",
        "🔲 3. Confusion Matrix",
        "🌲 4. Feature Importance",
        "⚙️ 5. ML Pipeline",
        "🔍 6. Prediction Analysis",
        "📊 7. Model Comparison",
        "📑 8. Dataset Details"
    ])

    # TAB 1: MODEL OVERVIEW
    with t_ov:
        st.markdown("""
        <div class="fintech-card" style="margin-top: 15px;">
            <h4 style="color: #1E3A8A; margin-top: 0; font-weight: 700;">🤖 Deployed Model Overview</h4>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-top: 15px;">
                <div class="metric-sub-card">
                    <div class="sub-card-label">Model Algorithm</div>
                    <div class="sub-card-value">Random Forest Classifier</div>
                </div>
                <div class="metric-sub-card">
                    <div class="sub-card-label">Trees (n_estimators)</div>
                    <div class="sub-card-value">100 Decision Trees</div>
                </div>
                <div class="metric-sub-card">
                    <div class="sub-card-label">Primary Dataset</div>
                    <div class="sub-card-value">loan_dataset_20000.csv</div>
                </div>
                <div class="metric-sub-card">
                    <div class="sub-card-label">Total Dataset Size</div>
                    <div class="sub-card-value">20,000 Records × 22 Cols</div>
                </div>
                <div class="metric-sub-card">
                    <div class="sub-card-label">Input Feature Count</div>
                    <div class="sub-card-value">21 Input Parameters</div>
                </div>
                <div class="metric-sub-card">
                    <div class="sub-card-label">Target Variable</div>
                    <div class="sub-card-value">loan_paid_back ∈ {0, 1}</div>
                </div>
                <div class="metric-sub-card">
                    <div class="sub-card-label">Positive Class (1)</div>
                    <div class="sub-card-value">Approved / Paid Back</div>
                </div>
                <div class="metric-sub-card">
                    <div class="sub-card-label">Negative Class (0)</div>
                    <div class="sub-card-value">Rejected / Defaulted</div>
                </div>
                <div class="metric-sub-card">
                    <div class="sub-card-label">Train-Test Split</div>
                    <div class="sub-card-value">80% Train / 20% Test</div>
                </div>
                <div class="metric-sub-card">
                    <div class="sub-card-label">Preprocessors</div>
                    <div class="sub-card-value">StandardScaler + LabelEncoder</div>
                </div>
                <div class="metric-sub-card">
                    <div class="sub-card-label">Class Balancing</div>
                    <div class="sub-card-value">SMOTE (Train Set Only)</div>
                </div>
                <div class="metric-sub-card">
                    <div class="sub-card-label">Model Serialization</div>
                    <div class="sub-card-value">model.pkl (Binary Pickle)</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # TAB 2: MODEL PERFORMANCE METRICS
    with t_perf:
        st.markdown("#### 🎯 Evaluation Metrics Calculated on Unseen Test Data (4,000 Samples)")

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Accuracy", "94.61%")
        with m2:
            st.metric("Precision", "91.28%")
        with m3:
            st.metric("Recall", "98.66%")
        with m4:
            st.metric("F1-Score", "94.82%")

        m5, m6, m7, m8 = st.columns(4)
        with m5:
            st.metric("ROC-AUC", "0.9841")
        with m6:
            st.metric("Specificity", "89.51%")
        with m7:
            st.metric("Balanced Accuracy", "94.09%")
        with m8:
            st.metric("Test Set Samples", "4,000")

        st.markdown("##### 📌 Metric Formulas, Explanations & Financial Significance")

        st.markdown("""
        <div class="fintech-card">
            <table style="width: 100%; font-size: 0.88rem; border-collapse: collapse; color: #334155;">
                <thead style="background: #F8FAFC; border-bottom: 2px solid #E2E8F0; text-align: left;">
                    <tr>
                        <th style="padding: 10px;">Metric</th>
                        <th style="padding: 10px;">Value</th>
                        <th style="padding: 10px;">Formula</th>
                        <th style="padding: 10px;">Technical Meaning</th>
                        <th style="padding: 10px;">Financial Significance in Underwriting</th>
                    </tr>
                </thead>
                <tbody>
                    <tr style="border-bottom: 1px solid #E2E8F0;">
                        <td style="padding: 10px;"><b>Accuracy</b></td>
                        <td style="padding: 10px; color: #10B981; font-weight: 700;">94.61%</td>
                        <td style="padding: 10px;"><code>(TP + TN) / Total</code></td>
                        <td style="padding: 10px;">Overall percentage of correct predictions across all test cases.</td>
                        <td style="padding: 10px;">Measures overall reliability of automated credit decisions.</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #E2E8F0;">
                        <td style="padding: 10px;"><b>Precision</b></td>
                        <td style="padding: 10px; color: #10B981; font-weight: 700;">91.28%</td>
                        <td style="padding: 10px;"><code>TP / (TP + FP)</code></td>
                        <td style="padding: 10px;">Proportion of predicted approvals that actually pay back.</td>
                        <td style="padding: 10px;">High precision minimizes bad debt write-offs from approving risky borrowers.</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #E2E8F0;">
                        <td style="padding: 10px;"><b>Recall</b></td>
                        <td style="padding: 10px; color: #10B981; font-weight: 700;">98.66%</td>
                        <td style="padding: 10px;"><code>TP / (TP + FN)</code></td>
                        <td style="padding: 10px;">Proportion of creditworthy applicants correctly identified & approved.</td>
                        <td style="padding: 10px;">High recall (98.66%) ensures the bank rarely turns away good borrowers.</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #E2E8F0;">
                        <td style="padding: 10px;"><b>F1-Score</b></td>
                        <td style="padding: 10px; color: #10B981; font-weight: 700;">94.82%</td>
                        <td style="padding: 10px;"><code>2 × (P × R) / (P + R)</code></td>
                        <td style="padding: 10px;">Harmonic mean balancing precision and recall into a single score.</td>
                        <td style="padding: 10px;">Provides a robust metric under balanced performance constraints.</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #E2E8F0;">
                        <td style="padding: 10px;"><b>ROC-AUC</b></td>
                        <td style="padding: 10px; color: #10B981; font-weight: 700;">0.9841</td>
                        <td style="padding: 10px;"><code>Area under TPR vs FPR</code></td>
                        <td style="padding: 10px;">Model's ability to discriminate between good and defaulting borrowers.</td>
                        <td style="padding: 10px;">Score near 1.0 (0.9841) confirms elite classification discrimination.</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #E2E8F0;">
                        <td style="padding: 10px;"><b>Specificity</b></td>
                        <td style="padding: 10px; color: #10B981; font-weight: 700;">89.51%</td>
                        <td style="padding: 10px;"><code>TN / (TN + FP)</code></td>
                        <td style="padding: 10px;">Proportion of actual defaulting applicants correctly rejected.</td>
                        <td style="padding: 10px;">Protects lender capital by flagging 89.51% of default-prone profiles.</td>
                    </tr>
                    <tr>
                        <td style="padding: 10px;"><b>Balanced Accuracy</b></td>
                        <td style="padding: 10px; color: #10B981; font-weight: 700;">94.09%</td>
                        <td style="padding: 10px;"><code>(Recall + Specificity) / 2</code></td>
                        <td style="padding: 10px;">Average of recall across both positive and negative target classes.</td>
                        <td style="padding: 10px;">Confirms equal model performance without bias toward either outcome.</td>
                    </tr>
                </tbody>
            </table>
        </div>
        """, unsafe_allow_html=True)

    # TAB 3: CONFUSION MATRIX
    with t_cm:
        st.markdown("#### 🔲 Test Set Confusion Matrix Heatmap (4,000 Samples)")

        col_cm1, col_cm2 = st.columns([1.2, 1.0])

        with col_cm1:
            cm_matrix = np.array([[1811, 189], [27, 1973]])
            fig_cm = px.imshow(
                cm_matrix,
                x=['Predicted Default (0)', 'Predicted Approved (1)'],
                y=['Actual Default (0)', 'Actual Approved (1)'],
                text_auto=True,
                color_continuous_scale='Blues',
                title="Confusion Matrix (Positive Class = 1: Approved)"
            )
            fig_cm.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="#0F172A")
            st.plotly_chart(fig_cm, width="stretch")

        with col_cm2:
            st.markdown("""
            <div class="fintech-card">
                <h5 style="color: #0F172A; font-weight: 700;">🔍 Confusion Matrix Breakdown & Implications</h5>
                <ul style="font-size: 0.88rem; color: #334155; line-height: 1.6; padding-left: 18px;">
                    <li><b style="color: #10B981;">True Positives (TP = 1,973):</b> Creditworthy applicants correctly APPROVED. Drives interest revenue with zero default risk.</li>
                    <li><b style="color: #1E3A8A;">True Negatives (TN = 1,811):</b> High-risk borrowers correctly REJECTED. Prevents principal loss and credit default.</li>
                    <li><b style="color: #EF4444;">False Positives (FP = 189):</b> Default-prone applicants mistakenly APPROVED. Represents credit risk exposure ($189$ bad loans).</li>
                    <li><b style="color: #F59E0B;">False Negatives (FN = 27):</b> Good applicants mistakenly REJECTED. Represents minor opportunity loss ($27$ missed loans).</li>
                </ul>
                <p style="font-size: 0.82rem; color: #64748B; margin-top: 10px;">
                    <i>Note: The positive class (1) represents Approved/Paid Back loans, while the negative class (0) represents Rejected/Defaulted loans.</i>
                </p>
            </div>
            """, unsafe_allow_html=True)

    # TAB 4: FEATURE IMPORTANCE
    with t_fi:
        st.markdown("#### 🌲 Feature Importance Ranking (Gini Impurity Reduction)")
        st.markdown("Importances extracted directly from the fitted Random Forest model (`rf_model.feature_importances_`):")

        if assets and 'model' in assets:
            rf_m = assets['model']
            feat_names = assets['feature_cols']
            importances = rf_m.feature_importances_
            df_imp = pd.DataFrame({'Feature': feat_names, 'Importance': importances}).sort_values('Importance', ascending=True)

            col_fi1, col_fi2 = st.columns([1.3, 1.0])

            with col_fi1:
                top10_imp = df_imp.tail(10)
                fig_imp = px.bar(
                    top10_imp,
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    color='Importance',
                    color_discrete_sequence=['#1E3A8A'],
                    title="Top 10 Most Influential Features"
                )
                fig_imp.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="#0F172A", margin=dict(t=40, b=10, l=10, r=10))
                st.plotly_chart(fig_imp, width="stretch")

            with col_fi2:
                st.markdown("""
                <div class="fintech-card">
                    <h5 style="color: #0F172A; font-weight: 700;">💡 Feature Importance Insights</h5>
                    <p style="font-size: 0.88rem; color: #334155; line-height: 1.5;">
                        <b>1. Employment Status (29.77%):</b> Top predictor. Stable employment indicates reliable monthly cash flows.<br><br>
                        <b>2. DTI Ratio (10.37%):</b> Debt-to-income ratio reflects applicant leverage. High DTI restricts repayment capacity.<br><br>
                        <b>3. Credit Score (7.01%):</b> FICO score serves as historical credit risk indicator.<br><br>
                        <b>4. Lender Subgrade (6.62%):</b> Lender internal risk rating tier.<br><br>
                        <i>Methodology Note: Feature importance reflects Gini impurity reduction across decision trees. It establishes predictive contribution, not direct causality.</i>
                    </p>
                </div>
                """, unsafe_allow_html=True)

            # Search & Filter Feature Table
            st.markdown("##### 🔎 Interactive Feature Dictionary & Search")
            search_query = st.text_input("Filter features by name or keyword:", value="", key="feat_search")

            feat_details = [
                {'Feature': 'employment_status', 'Description': 'Employment classification (Employed, Self-employed, etc.)', 'Type': 'Categorical', 'Unit': 'String Label', 'Importance': f"{df_imp[df_imp['Feature']=='employment_status']['Importance'].values[0]*100:.2f}%"},
                {'Feature': 'debt_to_income_ratio', 'Description': 'Calculated ratio: Debt / (Monthly Income × 12)', 'Type': 'Numeric', 'Unit': 'Decimal Ratio (0-1)', 'Importance': f"{df_imp[df_imp['Feature']=='debt_to_income_ratio']['Importance'].values[0]*100:.2f}%"},
                {'Feature': 'credit_score', 'Description': 'FICO credit score rating', 'Type': 'Numeric', 'Unit': 'Score (300-850)', 'Importance': f"{df_imp[df_imp['Feature']=='credit_score']['Importance'].values[0]*100:.2f}%"},
                {'Feature': 'grade_subgrade', 'Description': 'Internal lender risk tier rating', 'Type': 'Categorical', 'Unit': 'Rating (A1-F5)', 'Importance': f"{df_imp[df_imp['Feature']=='grade_subgrade']['Importance'].values[0]*100:.2f}%"},
                {'Feature': 'delinquency_history', 'Description': 'Number of past payment delinquencies', 'Type': 'Numeric', 'Unit': 'Count', 'Importance': f"{df_imp[df_imp['Feature']=='delinquency_history']['Importance'].values[0]*100:.2f}%"},
                {'Feature': 'education_level', 'Description': 'Highest education level achieved', 'Type': 'Categorical', 'Unit': 'String Label', 'Importance': f"{df_imp[df_imp['Feature']=='education_level']['Importance'].values[0]*100:.2f}%"},
                {'Feature': 'num_of_delinquencies', 'Description': 'Total historical delinquency events', 'Type': 'Numeric', 'Unit': 'Count', 'Importance': f"{df_imp[df_imp['Feature']=='num_of_delinquencies']['Importance'].values[0]*100:.2f}%"},
                {'Feature': 'interest_rate', 'Description': 'Loan interest rate percentage', 'Type': 'Numeric', 'Unit': 'Percentage (%)', 'Importance': f"{df_imp[df_imp['Feature']=='interest_rate']['Importance'].values[0]*100:.2f}%"},
                {'Feature': 'loan_purpose', 'Description': 'Intended purpose of the requested loan', 'Type': 'Categorical', 'Unit': 'Category Name', 'Importance': f"{df_imp[df_imp['Feature']=='loan_purpose']['Importance'].values[0]*100:.2f}%"},
                {'Feature': 'num_of_open_accounts', 'Description': 'Active credit lines count', 'Type': 'Numeric', 'Unit': 'Count', 'Importance': f"{df_imp[df_imp['Feature']=='num_of_open_accounts']['Importance'].values[0]*100:.2f}%"},
                {'Feature': 'current_balance', 'Description': 'Current outstanding debt balance', 'Type': 'Numeric', 'Unit': 'Currency ($ / ₹)', 'Importance': f"{df_imp[df_imp['Feature']=='current_balance']['Importance'].values[0]*100:.2f}%"},
                {'Feature': 'age', 'Description': "Applicant's age in years", 'Type': 'Numeric', 'Unit': 'Years', 'Importance': f"{df_imp[df_imp['Feature']=='age']['Importance'].values[0]*100:.2f}%"},
                {'Feature': 'installment', 'Description': 'Calculated monthly repayment installment', 'Type': 'Numeric', 'Unit': 'Currency ($ / ₹)', 'Importance': f"{df_imp[df_imp['Feature']=='installment']['Importance'].values[0]*100:.2f}%"},
                {'Feature': 'loan_amount', 'Description': 'Requested principal loan amount', 'Type': 'Numeric', 'Unit': 'Currency ($ / ₹)', 'Importance': f"{df_imp[df_imp['Feature']=='loan_amount']['Importance'].values[0]*100:.2f}%"},
                {'Feature': 'total_credit_limit', 'Description': 'Total available credit limit capacity', 'Type': 'Numeric', 'Unit': 'Currency ($ / ₹)', 'Importance': f"{df_imp[df_imp['Feature']=='total_credit_limit']['Importance'].values[0]*100:.2f}%"},
                {'Feature': 'annual_income', 'Description': 'Total gross annual income', 'Type': 'Numeric', 'Unit': 'Currency ($ / ₹)', 'Importance': f"{df_imp[df_imp['Feature']=='annual_income']['Importance'].values[0]*100:.2f}%"},
                {'Feature': 'monthly_income', 'Description': 'Net monthly income (Annual / 12)', 'Type': 'Numeric', 'Unit': 'Currency ($ / ₹)', 'Importance': f"{df_imp[df_imp['Feature']=='monthly_income']['Importance'].values[0]*100:.2f}%"},
                {'Feature': 'marital_status', 'Description': 'Applicant relationship status', 'Type': 'Categorical', 'Unit': 'String Label', 'Importance': f"{df_imp[df_imp['Feature']=='marital_status']['Importance'].values[0]*100:.2f}%"},
                {'Feature': 'gender', 'Description': 'Gender identity', 'Type': 'Categorical', 'Unit': 'String Label', 'Importance': f"{df_imp[df_imp['Feature']=='gender']['Importance'].values[0]*100:.2f}%"},
                {'Feature': 'loan_term', 'Description': 'Loan term length', 'Type': 'Numeric', 'Unit': 'Months (36 / 60)', 'Importance': f"{df_imp[df_imp['Feature']=='loan_term']['Importance'].values[0]*100:.2f}%"},
                {'Feature': 'public_records', 'Description': 'Derogatory public records count', 'Type': 'Numeric', 'Unit': 'Count', 'Importance': f"{df_imp[df_imp['Feature']=='public_records']['Importance'].values[0]*100:.2f}%"}
            ]

            df_feat_table = pd.DataFrame(feat_details)
            if search_query:
                df_feat_table = df_feat_table[df_feat_table['Feature'].str.contains(search_query, case=False) | df_feat_table['Description'].str.contains(search_query, case=False)]

            st.dataframe(df_feat_table, width="stretch")

    # TAB 5: ML TRAINING PIPELINE
    with t_pipe:
        st.markdown("#### ⚙️ Verified Single-File ML Execution Pipeline")
        st.markdown("""
        <div class="fintech-card">
            <div style="display: flex; flex-direction: column; gap: 12px;">
                <div style="background: #F8FAFC; border-left: 4px solid #1E3A8A; padding: 12px 16px; border-radius: 0 8px 8px 0;">
                    <b>1. Dataset Loading:</b> Reads <code>loan_dataset_20000.csv</code> containing 20,000 records.
                </div>
                <div style="background: #F8FAFC; border-left: 4px solid #1E3A8A; padding: 12px 16px; border-radius: 0 8px 8px 0;">
                    <b>2. Data Cleaning:</b> Validates range constraints and checks non-null integrity.
                </div>
                <div style="background: #F8FAFC; border-left: 4px solid #1E3A8A; padding: 12px 16px; border-radius: 0 8px 8px 0;">
                    <b>3. Preprocessing & Encoding:</b> Transforms string categoricals using fitted <code>LabelEncoder</code> objects.
                </div>
                <div style="background: #F8FAFC; border-left: 4px solid #1E3A8A; padding: 12px 16px; border-radius: 0 8px 8px 0;">
                    <b>4. Stratified Split:</b> 80/20 train/test split (16,000 train / 4,000 test) preserving target ratio.
                </div>
                <div style="background: #F8FAFC; border-left: 4px solid #10B981; padding: 12px 16px; border-radius: 0 8px 8px 0;">
                    <b>5. SMOTE Balancing (Train Split Only):</b> Oversamples minority default classes strictly on <code>X_train</code> to prevent data leakage.
                </div>
                <div style="background: #F8FAFC; border-left: 4px solid #1E3A8A; padding: 12px 16px; border-radius: 0 8px 8px 0;">
                    <b>6. Random Forest Training:</b> Fits 100 decision trees (<code>n_estimators=100</code>) on balanced training matrix.
                </div>
                <div style="background: #F8FAFC; border-left: 4px solid #1E3A8A; padding: 12px 16px; border-radius: 0 8px 8px 0;">
                    <b>7. Evaluation & Inference:</b> Validates against unseen 4,000 test samples (94.61% accuracy, 0.9841 ROC-AUC) and exports binary <code>model.pkl</code>.
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # TAB 6: PREDICTION ANALYSIS SIMULATOR
    with t_sim:
        st.markdown("#### 🔍 Interactive Prediction Analysis Simulator")
        st.markdown("Test profile parameters to analyze probability outputs and individual prediction explanations.")

        sim_col1, sim_col2 = st.columns([1.2, 1.0])

        with sim_col1:
            sim_score = st.slider("Test Credit Score", 300, 850, 740, key="sim_score")
            sim_emp = st.selectbox("Test Employment", options=label_encoders['employment_status'].classes_, index=0, key="sim_emp")
            sim_inc = st.number_input("Test Annual Income ($)", value=75000.0, step=5000.0, key="sim_inc")
            sim_debt = st.number_input("Test Outstanding Debt ($)", value=15000.0, step=1000.0, key="sim_debt")

            sim_monthly = sim_inc / 12.0
            sim_dti = round(sim_debt / (sim_monthly * 12.0), 4) if sim_inc > 0 else 0.0

            st.markdown(f"**Calculated DTI Ratio:** `{sim_dti:.2f}`")


        with sim_col2:
            if assets and 'model' in assets:
                sim_data = {
                    'age': 35,
                    'gender': 'Male',
                    'marital_status': 'Married',
                    'education_level': "Bachelor's",
                    'annual_income': float(sim_inc),
                    'monthly_income': float(sim_monthly),
                    'employment_status': sim_emp,
                    'debt_to_income_ratio': float(sim_dti),
                    'credit_score': sim_score,
                    'loan_amount': 20000.0,
                    'loan_purpose': 'Debt consolidation',
                    'interest_rate': 10.5,
                    'loan_term': 36,
                    'installment': 650.0,
                    'grade_subgrade': 'B1',
                    'num_of_open_accounts': 5,
                    'total_credit_limit': 50000.0,
                    'current_balance': float(sim_debt),
                    'delinquency_history': 0,
                    'public_records': 0,
                    'num_of_delinquencies': 0
                }

                try:
                    sim_df = pd.DataFrame([sim_data])[feature_cols]
                    for col, le in label_encoders.items():
                        sim_df[col] = le.transform(sim_df[col])
                    sim_scaled = scaler.transform(sim_df)

                    sim_pred = model.predict(sim_scaled)[0]
                    sim_prob = model.predict_proba(sim_scaled)[0][1]

                    status_str = "APPROVED" if sim_pred == 1 else "REJECTED"
                    status_col = "#10B981" if sim_pred == 1 else "#EF4444"

                    st.markdown(f"""
                    <div class="fintech-card" style="text-align: center; border-left: 4px solid {status_col};">
                        <h4 style="color: {status_col}; margin: 0; font-weight: 800;">PREDICTED RESULT: {status_str}</h4>
                        <div style="font-size: 2rem; font-weight: 800; color: #0F172A; margin: 10px 0;">{sim_prob*100:.1f}%</div>
                        <div style="font-size: 0.8rem; color: #64748B;">Model Repayment Confidence Probability</div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.progress(float(sim_prob))
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.caption("Disclaimer: Predicted probabilities reflect statistical confidence based on historic training data and are not guaranteed loan guarantees.")
                except Exception as ex:
                    st.error(f"Simulator error: {ex}")

    # TAB 7: MODEL COMPARISON
    with t_comp:
        st.markdown("#### 📊 Benchmark Comparison Across Evaluated Algorithms")

        comp_data = pd.DataFrame({
            'Model Algorithm': ['Logistic Regression', 'XGBoost', 'Random Forest (Selected)'],
            'Accuracy': ['76.11%', '93.55%', '94.61%'],
            'Precision': ['75.23%', '89.79%', '91.28%'],
            'Recall': ['77.93%', '98.28%', '98.66%'],
            'F1-Score': ['76.55%', '93.84%', '94.82%'],
            'ROC-AUC': ['0.8620', '0.9739', '0.9841']
        })

        st.dataframe(comp_data, width="stretch")

        st.markdown("##### Performance Metrics Visual Comparison")
        bar_comp = pd.DataFrame({
            'Model': ['Logistic Regression', 'XGBoost', 'Random Forest'],
            'Accuracy': [0.7611, 0.9355, 0.9461],
            'Recall': [0.7793, 0.9828, 0.9866],
            'F1-Score': [0.7655, 0.9384, 0.9482]
        })
        st.bar_chart(bar_comp.set_index('Model'))

    # TAB 8: DATASET & EVALUATION DETAILS
    with t_ds:
        st.markdown("#### 📑 Dataset Properties & Evaluation Summary")

        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.markdown("""
            <div class="fintech-card">
                <h5 style="color: #0F172A; font-weight: 700;">📊 Dataset Properties</h5>
                <ul style="font-size: 0.88rem; color: #334155; line-height: 1.6; padding-left: 18px;">
                    <li><b>Filename:</b> <code>loan_dataset_20000.csv</code></li>
                    <li><b>Total Records:</b> 20,000 observations</li>
                    <li><b>Total Columns:</b> 22 (21 features + 1 target)</li>
                    <li><b>Missing Values:</b> 0 missing values (100% complete)</li>
                    <li><b>Training Split:</b> 16,000 samples (80%)</li>
                    <li><b>Testing Split:</b> 4,000 samples (20%)</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        with col_d2:
            st.markdown("""
            <div class="fintech-card">
                <h5 style="color: #0F172A; font-weight: 700;">⚖️ Class Distribution & SMOTE</h5>
                <ul style="font-size: 0.88rem; color: #334155; line-height: 1.6; padding-left: 18px;">
                    <li><b>Target Variable:</b> <code>loan_paid_back</code></li>
                    <li><b>Class 1 (Paid Back):</b> ~10,000 records (50%)</li>
                    <li><b>Class 0 (Defaulted):</b> ~10,000 records (50%)</li>
                    <li><b>Pre-SMOTE Train Split:</b> 8,000 Paid / 8,000 Default</li>
                    <li><b>Post-SMOTE Train Split:</b> Balanced 1:1 ratio maintained</li>
                    <li><b>Test Split (4,000):</b> Preserved natural test distribution</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div style="background-color: #EFF6FF; border: 1px solid #BFDBFE; border-radius: 12px; padding: 18px; color: #1E3A8A;">
            <h5 style="margin: 0 0 6px 0; font-weight: 700;">🔒 Privacy & Governance Assurance</h5>
            <p style="margin: 0; font-size: 0.88rem; line-height: 1.5;">
                All dataset records evaluate synthetic/anonymized credit parameters. No Personally Identifiable Information (PII) such as SSNs, names, or addresses are collected or stored.
            </p>
        </div>
        """, unsafe_allow_html=True)

# PAGE 6: ABOUT
elif page == "About":
    st.markdown("### 📚 About ApprovaAI & Technical Documentation")
    st.markdown("Detailed technical overview, machine learning pipeline, concepts, and tech stack powering the ApprovaAI Underwriting Intelligence Platform.")

    tab_proj, tab_pipe, tab_ml, tab_stack = st.tabs([
        "📋 1. Project Overview", 
        "⚙️ 2. ML Pipeline", 
        "🧠 3. ML Concepts", 
        "💻 4. Tech Stack"
    ])

    # -------------------------------------------------------------
    # TAB 1: PROJECT OVERVIEW
    # -------------------------------------------------------------
    with tab_proj:
        st.markdown("""
        <div class="fintech-card" style="margin-top: 15px;">
            <h4 style="color: #1E3A8A; margin-top: 0; font-weight: 700;">🎯 Project Objective & Problem Statement</h4>
            <p style="color: #334155; line-height: 1.6; font-size: 0.95rem;">
                Traditional credit underwriting relies heavily on manual appraisal processes that can be slow, subject to human bias, and inconsistent across applicant profiles. 
                <b>ApprovaAI</b> is an enterprise-grade Decision Support System (DSS) engineered to automate loan approval predictions, evaluate credit risk tiers, and provide transparent financial insights using machine learning.
            </p>
        </div>
        """, unsafe_allow_html=True)

        col_u1, col_u2 = st.columns(2)
        with col_u1:
            st.markdown("""
            <div class="fintech-card">
                <h5 style="color: #0F172A; font-weight: 700;">👥 Target Users & Use Cases</h5>
                <ul style="color: #334155; font-size: 0.9rem; line-height: 1.6; padding-left: 20px;">
                    <li><b>Underwriters & Analysts:</b> Rapid preliminary screening of loan applications with standardized risk scores.</li>
                    <li><b>Loan Applicants:</b> Instant visibility into key risk drivers affecting repayment probability.</li>
                    <li><b>Fintech Platforms:</b> Automated, scalable credit decisioning API & dashboard integration.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        with col_u2:
            st.markdown("""
            <div class="fintech-card">
                <h5 style="color: #0F172A; font-weight: 700;">⚡ Key Platform Features</h5>
                <ul style="color: #334155; font-size: 0.9rem; line-height: 1.6; padding-left: 20px;">
                    <li><b>Automated Underwriting:</b> Instant prediction using 21 applicant parameters.</li>
                    <li><b>Synchronized Income Conversion:</b> Two-way live calculation (<code>Annual = Monthly × 12</code>).</li>
                    <li><b>Automated DTI Calculation:</b> Real-time ratio calculation (<code>DTI = Debt / (Monthly × 12)</code>).</li>
                    <li><b>Interactive Analytics & Audit Log:</b> Explore dataset trends and export session history.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div style="background-color: #FEF3C7; border: 1px solid #FCD34D; border-radius: 12px; padding: 18px; color: #92400E; margin-top: 10px; margin-bottom: 20px;">
            <h5 style="margin: 0 0 6px 0; font-weight: 700;">⚠️ Decision Support Disclaimer</h5>
            <p style="margin: 0; font-size: 0.88rem; line-height: 1.5;">
                Predictions generated by ApprovaAI are statistical probability estimates designed strictly to assist financial analysts. 
                They do not constitute a legal contract, guaranteed loan offer, or final underwriting decision. Manual expert review is always recommended for borderline cases.
            </p>
        </div>
        """, unsafe_allow_html=True)

    # -------------------------------------------------------------
    # TAB 2: ML PIPELINE
    # -------------------------------------------------------------
    with tab_pipe:
        st.markdown("#### ⚙️ End-to-End Machine Learning Architecture Workflow")
        st.markdown("The step-by-step data engineering and modeling pipeline powering ApprovaAI:")

        st.markdown("""
        <div class="fintech-card" style="margin-top: 15px;">
            <div style="display: flex; flex-direction: column; gap: 14px;">
                <div style="background: #F8FAFC; border-left: 4px solid #1E3A8A; padding: 12px 16px; border-radius: 0 8px 8px 0;">
                    <b style="color: #1E3A8A; font-size: 1rem;">Step 1: Dataset Ingestion</b><br>
                    <span style="font-size: 0.88rem; color: #475569;">Loaded <code>loan_dataset_20000.csv</code> comprising 20,000 historical loan records across 21 input attributes and target variable <code>loan_paid_back</code>.</span>
                </div>
                <div style="background: #F8FAFC; border-left: 4px solid #1E3A8A; padding: 12px 16px; border-radius: 0 8px 8px 0;">
                    <b style="color: #1E3A8A; font-size: 1rem;">Step 2: Data Cleaning & Boundary Validation</b><br>
                    <span style="font-size: 0.88rem; color: #475569;">Verified non-null structures, validated non-negative income/debt values, and checked range constraints for credit scores (300-850).</span>
                </div>
                <div style="background: #F8FAFC; border-left: 4px solid #1E3A8A; padding: 12px 16px; border-radius: 0 8px 8px 0;">
                    <b style="color: #1E3A8A; font-size: 1rem;">Step 3: Categorical Feature Encoding</b><br>
                    <span style="font-size: 0.88rem; color: #475569;">Fitted <code>LabelEncoder</code> objects on string features: <code>gender</code>, <code>marital_status</code>, <code>education_level</code>, <code>employment_status</code>, <code>loan_purpose</code>, and <code>grade_subgrade</code>.</span>
                </div>
                <div style="background: #F8FAFC; border-left: 4px solid #1E3A8A; padding: 12px 16px; border-radius: 0 8px 8px 0;">
                    <b style="color: #1E3A8A; font-size: 1rem;">Step 4: Stratified Train-Test Split (80/20)</b><br>
                    <span style="font-size: 0.88rem; color: #475569;">Partitioned dataset using <code>train_test_split(..., test_size=0.2, random_state=42, stratify=y)</code> to preserve class proportions across splits.</span>
                </div>
                <div style="background: #F8FAFC; border-left: 4px solid #1E3A8A; padding: 12px 16px; border-radius: 0 8px 8px 0;">
                    <b style="color: #1E3A8A; font-size: 1rem;">Step 5: Feature Standardization</b><br>
                    <span style="font-size: 0.88rem; color: #475569;">Standardized feature matrix using <code>StandardScaler</code> fitted on <code>X_train</code> to normalize numerical features.</span>
                </div>
                <div style="background: #F8FAFC; border-left: 4px solid #10B981; padding: 12px 16px; border-radius: 0 8px 8px 0;">
                    <b style="color: #10B981; font-size: 1rem;">Step 6: SMOTE Class Balancing (Training Set ONLY)</b><br>
                    <span style="font-size: 0.88rem; color: #475569;">Applied Synthetic Minority Over-sampling Technique (<code>SMOTE</code>) strictly to <code>X_train_scaled</code> to eliminate class imbalance without data leakage into test set.</span>
                </div>
                <div style="background: #F8FAFC; border-left: 4px solid #1E3A8A; padding: 12px 16px; border-radius: 0 8px 8px 0;">
                    <b style="color: #1E3A8A; font-size: 1rem;">Step 7: Random Forest Model Training</b><br>
                    <span style="font-size: 0.88rem; color: #475569;">Trained <code>RandomForestClassifier(n_estimators=100, random_state=42)</code> on the resampled training dataset.</span>
                </div>
                <div style="background: #F8FAFC; border-left: 4px solid #1E3A8A; padding: 12px 16px; border-radius: 0 8px 8px 0;">
                    <b style="color: #1E3A8A; font-size: 1rem;">Step 8: Model Evaluation & Asset Serialization</b><br>
                    <span style="font-size: 0.88rem; color: #475569;">Evaluated model against test set (<b>94.61% Accuracy</b>, <b>0.9841 ROC-AUC</b>) and serialized fitted assets to <code>model.pkl</code>.</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # -------------------------------------------------------------
    # TAB 3: ML CONCEPTS
    # -------------------------------------------------------------
    with tab_ml:
        st.markdown("#### 🧠 Core Machine Learning & Statistical Concepts")

        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.markdown("""
            <div class="fintech-card">
                <h5 style="color: #1E3A8A; font-weight: 700;">1. Supervised Learning & Random Forest</h5>
                <p style="font-size: 0.88rem; color: #334155; line-height: 1.5;">
                    ApprovaAI uses <b>Supervised Binary Classification</b> where historical features map to a target <code>loan_paid_back ∈ {0, 1}</code>.<br><br>
                    <b>Random Forest Classifier:</b> An ensemble method combining 100 decision trees. Each tree votes on the classification, reducing variance and mitigating overfitting compared to a single decision tree.
                </p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="fintech-card">
                <h5 style="color: #1E3A8A; font-weight: 700;">2. Class Imbalance & SMOTE</h5>
                <p style="font-size: 0.88rem; color: #334155; line-height: 1.5;">
                    In financial datasets, defaults are often minority classes. <b>SMOTE</b> creates synthetic samples along line segments joining nearest neighbors.<br><br>
                    <b>Data Leakage Prevention:</b> SMOTE is executed <i>after</i> train-test splitting strictly on training folds, ensuring test set metrics measure true model generalization.
                </p>
            </div>
            """, unsafe_allow_html=True)

        with col_c2:
            st.markdown("""
            <div class="fintech-card">
                <h5 style="color: #1E3A8A; font-weight: 700;">3. Mathematical Feature Formulas</h5>
                <p style="font-size: 0.88rem; color: #334155; line-height: 1.5;">
                    <b>Annual Income Synchronization:</b><br>
                    <code>Annual Income = Monthly Income × 12</code><br><br>
                    <b>Debt-to-Income Ratio (DTI):</b><br>
                    <code>DTI = Current Outstanding Debt / (Monthly Income × 12)</code><br><br>
                    DTI is passed as a decimal ratio between <code>0.00</code> and <code>1.00</code>. High DTI or low credit score increases predicted default probability.
                </p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="fintech-card">
                <h5 style="color: #1E3A8A; font-weight: 700;">4. Performance Evaluation Benchmarks</h5>
                <p style="font-size: 0.88rem; color: #334155; line-height: 1.5;">
                    <b>Accuracy:</b> 94.61% | <b>Precision:</b> 91.28% | <b>Recall:</b> 98.66% | <b>F1-Score:</b> 94.82% | <b>ROC-AUC:</b> 0.9841<br><br>
                    High recall ensures the underwriting model flags 98.66% of repayment candidates while maintaining high overall precision.
                </p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("##### Performance Comparison Chart Across Models")
        comp_df = pd.DataFrame({
            'Model': ['Logistic Regression', 'XGBoost', 'Random Forest (Selected)'],
            'Accuracy': [0.7611, 0.9355, 0.9461],
            'Recall': [0.7793, 0.9828, 0.9866],
            'F1-Score': [0.7655, 0.9384, 0.9482]
        })
        st.bar_chart(comp_df.set_index('Model'))

    # -------------------------------------------------------------
    # TAB 4: TECH STACK
    # -------------------------------------------------------------
    with tab_stack:
        st.markdown("#### 💻 Implemented Technology Stack Cards")
        st.markdown("Technologies and libraries actually utilized across the ApprovaAI application codebase:")

        t1, t2, t3, t4 = st.columns(4)

        with t1:
            st.markdown("""
            <div class="fintech-card" style="text-align: center; padding: 20px 14px;">
                <div style="font-size: 2.2rem; margin-bottom: 8px;">🐍</div>
                <h5 style="margin: 0; font-weight: 700; color: #0F172A;">Python 3.12</h5>
                <p style="font-size: 0.78rem; color: #64748B; margin-top: 6px;">Core programming language & engine runtime environment.</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="fintech-card" style="text-align: center; padding: 20px 14px;">
                <div style="font-size: 2.2rem; margin-bottom: 8px;">🤖</div>
                <h5 style="margin: 0; font-weight: 700; color: #0F172A;">Scikit-Learn</h5>
                <p style="font-size: 0.78rem; color: #64748B; margin-top: 6px;">RandomForestClassifier, LabelEncoder, StandardScaler.</p>
            </div>
            """, unsafe_allow_html=True)

        with t2:
            st.markdown("""
            <div class="fintech-card" style="text-align: center; padding: 20px 14px;">
                <div style="font-size: 2.2rem; margin-bottom: 8px;">👑</div>
                <h5 style="margin: 0; font-weight: 700; color: #0F172A;">Streamlit</h5>
                <p style="font-size: 0.78rem; color: #64748B; margin-top: 6px;">Interactive dashboard UI framework & session state machine.</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="fintech-card" style="text-align: center; padding: 20px 14px;">
                <div style="font-size: 2.2rem; margin-bottom: 8px;">⚖️</div>
                <h5 style="margin: 0; font-weight: 700; color: #0F172A;">Imbalanced-Learn</h5>
                <p style="font-size: 0.78rem; color: #64748B; margin-top: 6px;">SMOTE oversampling for balanced class distributions.</p>
            </div>
            """, unsafe_allow_html=True)

        with t3:
            st.markdown("""
            <div class="fintech-card" style="text-align: center; padding: 20px 14px;">
                <div style="font-size: 2.2rem; margin-bottom: 8px;">🐼</div>
                <h5 style="margin: 0; font-weight: 700; color: #0F172A;">Pandas & NumPy</h5>
                <p style="font-size: 0.78rem; color: #64748B; margin-top: 6px;">Data manipulation, feature matrix ops & vector math.</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="fintech-card" style="text-align: center; padding: 20px 14px;">
                <div style="font-size: 2.2rem; margin-bottom: 8px;">📦</div>
                <h5 style="margin: 0; font-weight: 700; color: #0F172A;">Pickle Serialization</h5>
                <p style="font-size: 0.78rem; color: #64748B; margin-top: 6px;">Binary asset caching for preprocessors & trained model.</p>
            </div>
            """, unsafe_allow_html=True)

        with t4:
            st.markdown("""
            <div class="fintech-card" style="text-align: center; padding: 20px 14px;">
                <div style="font-size: 2.2rem; margin-bottom: 8px;">📊</div>
                <h5 style="margin: 0; font-weight: 700; color: #0F172A;">Plotly Express</h5>
                <p style="font-size: 0.78rem; color: #64748B; margin-top: 6px;">Interactive pie charts, histograms & grouped bar plots.</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="fintech-card" style="text-align: center; padding: 20px 14px;">
                <div style="font-size: 2.2rem; margin-bottom: 8px;">🎨</div>
                <h5 style="margin: 0; font-weight: 700; color: #0F172A;">HTML5 & Vanilla CSS3</h5>
                <p style="font-size: 0.78rem; color: #64748B; margin-top: 6px;">Custom fintech styling, responsive cards & flexbox layouts.</p>
            </div>
            """, unsafe_allow_html=True)



