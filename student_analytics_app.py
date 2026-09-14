import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==========================================
# 1. PAGE CONFIGURATION & STYLING
# ==========================================
st.set_page_config(
    page_title="USIU-Africa Student Analytics",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling for a clean, academic theme
st.markdown("""
    <style>
    .main {
        padding-top: 1.5rem;
    }
    .stMetric {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #e9ecef;
    }
    .stAlert {
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. LOAD TRAINED ARTIFACTS
# ==========================================
@st.cache_resource
def load_artifacts():
    """Load pre-trained LightGBM model, column transformer, and target label encoder."""
    try:
        model = joblib.load("logistic_regression_model.pkl")  # Production LightGBM model
        preprocessor = joblib.load("student_preprocessor.pkl")
        label_encoder = joblib.load("target_label_encoder.pkl")
        return model, preprocessor, label_encoder
    except Exception as e:
        st.error(f"Error loading model artifacts: {e}")
        st.stop()

model, preprocessor, label_encoder = load_artifacts()

# ==========================================
# 3. HEADER & DASHBOARD OVERVIEW
# ==========================================
st.title("🎓 USIU-Africa Student Engagement & Performance Analytics")
st.markdown("""
Predict student academic outcomes (*Distinction, Pass, Fail, Withdrawn*) using 
demographic metadata and Virtual Learning Environment (VLE) engagement indicators.
""")

st.divider()

# ==========================================
# 4. SIDEBAR - STUDENT PROFILE INPUTS
# ==========================================
st.sidebar.header(" Student Profile Inputs")

st.sidebar.subheader("Demographics & Registration")
gender = st.sidebar.selectbox("Gender", ["M", "F"])
region = st.sidebar.selectbox(
    "Region", 
    ["East Anglian Region", "Scotland", "North Western Region", "South Region", 
     "South East Region", "West Midlands Region", "Wales", "North Region", 
     "South West Region", "London Region", "East Midlands Region", "Yorkshire Region", "Ireland"]
)
highest_education = st.sidebar.selectbox(
    "Highest Education Level", 
    ["HE Qualification", "A Level or Equivalent", "Lower Than A Level", "Post Graduate Qualification", "No Formal Quals"]
)
imd_band = st.sidebar.selectbox(
    "Deprivation Band (IMD)", 
    ["0-10%", "10-20%", "20-30%", "30-40%", "40-50%", "50-60%", "60-70%", "70-80%", "80-90%", "90-100%", "Unknown"]
)
age_band = st.sidebar.selectbox("Age Band", ["0-35", "35-55", "55<="])
disability = st.sidebar.selectbox("Disability Status", ["N", "Y"])

st.sidebar.subheader("Academic History")
num_of_prev_attempts = st.sidebar.number_input("Previous Module Attempts", min_value=0, max_value=10, value=0)
studied_credits = st.sidebar.number_input("Total Studied Credits", min_value=30, max_value=300, value=60, step=30)
date_registration = st.sidebar.number_input("Days Registered Before Start (Negative = Early)", value=-30, step=1)

st.sidebar.subheader("Online VLE & Course Performance")
total_weighted_score = st.sidebar.slider("Total Weighted Assessment Score (%)", min_value=0.0, max_value=100.0, value=65.0)
assessments_completed = st.sidebar.number_input("Assessments Completed", min_value=0, max_value=20, value=4)
total_clicks = st.sidebar.number_input("Total VLE Interactions (Clicks)", min_value=0, max_value=20000, value=1200, step=50)
active_days = st.sidebar.number_input("Total Active Days on VLE", min_value=0, max_value=250, value=45, step=1)

# Assemble input dataframe matching training structure
input_dict = {
    "gender": gender,
    "region": region,
    "highest_education": highest_education,
    "imd_band": imd_band,
    "age_band": age_band,
    "num_of_prev_attempts": num_of_prev_attempts,
    "studied_credits": studied_credits,
    "disability": disability,
    "date_registration": date_registration,
    "total_weighted_score": total_weighted_score,
    "assessments_completed": assessments_completed,
    "total_clicks": total_clicks,
    "active_days": active_days
}

input_df = pd.DataFrame([input_dict])

# ==========================================
# 5. MODEL PREDICTION & PROBABILITIES
# ==========================================
col_main, col_stats = st.columns([2, 1])

with col_main:
    st.subheader(" Outcome Prediction")
    
    # Preprocess inputs
    try:
        input_processed = preprocessor.transform(input_df)
        prediction_num = model.predict(input_processed)[0]
        prediction_label = label_encoder.inverse_transform([prediction_num])[0]
        probabilities = model.predict_proba(input_processed)[0]
    except Exception as e:
        st.error(f"Error during preprocessing or prediction: {e}")
        st.stop()
    
    # Display primary outcome badge
    if prediction_label == "Distinction":
        st.success(f"### Predicted Outcome: **{prediction_label}** ")
    elif prediction_label == "Pass":
        st.info(f"### Predicted Outcome: **{prediction_label}** ")
    elif prediction_label == "Fail":
        st.warning(f"### Predicted Outcome: **{prediction_label}** ")
    else:  # Withdrawn
        st.error(f"### Predicted Outcome: **{prediction_label}** ")

    # Display prediction probability distribution
    st.markdown("**Probability Breakdown Across Outcomes:**")
    prob_df = pd.DataFrame({
        "Outcome": label_encoder.classes_,
        "Probability (%)": (probabilities * 100).round(2)
    }).sort_values(by="Probability (%)", ascending=False)

    st.dataframe(
        prob_df, 
        column_config={
            "Probability (%)": st.column_config.ProgressColumn(
                "Probability (%)",
                format="%.2f%%",
                min_value=0,
                max_value=100,
            )
        },
        use_container_width=True,
        hide_index=True
    )

with col_stats:
    st.subheader(" Key Engagement Metrics")
    st.metric("VLE Active Days", f"{active_days} days")
    st.metric("Total VLE Interactions", f"{total_clicks:,} clicks")
    st.metric("Weighted Score", f"{total_weighted_score:.1f}%")

# ==========================================
# 6. FEATURE IMPORTANCE EXPLANATIONS
# ==========================================
st.divider()
st.subheader(" Key Drivers Behind Prediction")

try:
    feat_names = list(preprocessor.get_feature_names_out())
except AttributeError:
    feat_names = list(input_df.columns)

# Clean feature names for clear display
feat_names = [f.replace("num__", "").replace("cat__", "") for f in feat_names]

if hasattr(model, "feature_importances_"):
    importances = model.feature_importances_
    feat_imp = pd.DataFrame({
        "Feature": feat_names,
        "Relative Importance Score": importances
    }).sort_values(by="Relative Importance Score", ascending=False)

    st.markdown("Top global features influencing this model's decision boundary:")
    st.dataframe(
        feat_imp.head(7), 
        use_container_width=True,
        hide_index=True
    )