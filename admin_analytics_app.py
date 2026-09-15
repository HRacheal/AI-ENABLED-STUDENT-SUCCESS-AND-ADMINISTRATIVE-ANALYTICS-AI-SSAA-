import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set page layout and title
st.set_page_config(
    page_title="Administrative Efficiency Analytics",
    page_icon="🏢",
    layout="wide"
)

# Load trained models and preprocessors securely
@st.cache_resource
def load_artifacts():
    model = joblib.load("admin_resolution_model.pkl")
    preprocessor = joblib.load("admin_preprocessor.pkl")
    kmeans = joblib.load("admin_kmeans_cluster.pkl")
    cluster_scaler = joblib.load("admin_cluster_scaler.pkl")
    return model, preprocessor, kmeans, cluster_scaler

try:
    model, preprocessor, kmeans, cluster_scaler = load_artifacts()
except Exception as e:
    st.error("Error loading model artifacts. Please ensure all `.pkl` files are present in the app directory.")
    st.stop()

# Dashboard Header
st.title(" Administrative Efficiency Analytics System")
st.caption("AI-SSAA Component 3: Service Request Resolution Time Prediction & Workflow Bottleneck Identification")
st.divider()

# Sidebar: Input Form for New Service Request
st.sidebar.header(" New Service Request Details")

dept = st.sidebar.selectbox(
    "Department", 
    ["Registrar", "Finance", "IT", "Admissions", "Student Affairs", "Academic Affairs", "Library"]
)

req_type = st.sidebar.selectbox(
    "Request Type", 
    ["Transcript Request", "Payment Issue", "Portal Access", "Application Issue", 
     "Exam or Assessment Issue", "Password Reset", "Course Registration", "Fee Statement"]
)

priority = st.sidebar.selectbox(
    "Priority Level", 
    ["Low", "Medium", "High", "Critical"]
)

channel = st.sidebar.selectbox(
    "Communication Channel", 
    ["Portal", "Email", "Phone", "Walk-In"]
)

first_resp = st.sidebar.number_input(
    "First Response Time (Hours)", 
    min_value=0.1, max_value=24.0, value=4.5, step=0.5
)

interactions = st.sidebar.slider(
    "Number of Interactions", 
    min_value=1, max_value=15, value=3
)

escalated = st.sidebar.selectbox(
    "Escalated Ticket?", 
    [0, 1], 
    format_func=lambda x: "Yes" if x == 1 else "No"
)

# Format user input into a DataFrame for model inference
input_data = pd.DataFrame([{
    "Department": dept,
    "Request_Type": req_type,
    "Priority": priority,
    "Channel": channel,
    "First_Response_Time": first_resp,
    "Number_of_Interactions": interactions,
    "Escalated": escalated
}])

# Main View: Predictions & Clustering Analysis
col1, col2 = st.columns(2)

with col1:
    st.subheader("⏱ Resolution Time Prediction")
    
    # Preprocess inputs and predict turnaround time
    processed_input = preprocessor.transform(input_data)
    predicted_hours = model.predict(processed_input)[0]
    
    st.metric(
        label="Estimated Resolution Time", 
        value=f"{predicted_hours:.2f} Hours"
    )
    
    st.info(
        f"Based on historical data, a **{priority}** priority **{req_type}** ticket for the "
        f"**{dept}** department is expected to take ~**{predicted_hours:.2f} hours** to resolve."
    )

with col2:
    st.subheader(" Operational Risk Cluster")
    
    # Map input features to cluster space (Estimating baseline satisfaction score of 3.0)
    cluster_features = np.array([[first_resp, predicted_hours, 3.0]])
    scaled_cluster_input = cluster_scaler.transform(cluster_features)
    assigned_cluster = kmeans.predict(scaled_cluster_input)[0]
    
    cluster_profiles = {
        0: {"name": "Cluster 0: Medium-Complexity", "desc": "Standard workflow experiencing mild internal processing lag.", "color": "warning"},
        1: {"name": "Cluster 1: High-Complexity Bottleneck", "desc": "Delayed response & high interaction count requiring priority routing.", "color": "error"},
        2: {"name": "Cluster 2: Low-Complexity Fast Track", "desc": "Rapid initial response with low turnaround time and high satisfaction.", "color": "success"}
    }
    
    profile = cluster_profiles.get(assigned_cluster, {"name": f"Cluster {assigned_cluster}", "desc": "Standard workflow.", "color": "info"})
    
    if profile["color"] == "success":
        st.success(f"**{profile['name']}**\n\n{profile['desc']}")
    elif profile["color"] == "warning":
        st.warning(f"**{profile['name']}**\n\n{profile['desc']}")
    else:
        st.error(f"**{profile['name']}**\n\n{profile['desc']}")