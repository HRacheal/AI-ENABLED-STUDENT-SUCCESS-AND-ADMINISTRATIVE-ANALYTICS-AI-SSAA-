import streamlit as st
import json
import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="AI Virtual Advising Assistant", page_icon="🤖", layout="wide")

st.title("🤖 AI Student Support & Virtual Advising Assistant")
st.caption("AI-SSAA Component 2: 24/7 Automated University Administrative & Academic Support")
st.divider()

# Load saved model artifacts and original JSON
@st.cache_resource
def load_advising_resources():
    vectorizer = joblib.load("advising_vectorizer.pkl")
    df = joblib.load("advising_df.pkl")
    with open("intents.json", "r") as f:
        intents_data = json.load(f)
    return vectorizer, df, intents_data

try:
    vectorizer, df, intents_data = load_advising_resources()
    pattern_vectors = vectorizer.transform(df["pattern"].values)
except Exception as e:
    st.error("Error loading model artifacts. Ensure 'advising_vectorizer.pkl', 'advising_df.pkl', and 'intents.json' are in the root directory.")
    st.stop()

# Initialize Chat Memory
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am your Virtual Academic Advisor. Ask me anything about university fees, campus locations, operational hours, or department contacts."}
    ]

# Display Chat History
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"], unsafe_allow_html=True)

# User Query Prompt
if prompt := st.chat_input("Type your question here (e.g., How much do I pay for tuition fees?):"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Compute Cosine Similarity
    user_vec = vectorizer.transform([prompt])
    similarities = cosine_similarity(user_vec, pattern_vectors).flatten()
    best_idx = np.argmax(similarities)
    best_score = similarities[best_idx]

    # Threshold Matching Logic (0.20 confidence minimum)
    if best_score >= 0.20:
        matched_tag = df.iloc[best_idx]["tag"]
        matched_intent = next((item for item in intents_data["intents"] if item["tag"] == matched_tag), None)
        
        if matched_intent:
            bot_reply = np.random.choice(matched_intent["responses"])
        else:
            bot_reply = "I identified your topic, but couldn't retrieve a specific answer. Please contact student services."
    else:
        bot_reply = "I'm not sure I fully understand your request. I have logged this query for a human academic advisor to follow up with you."

    st.session_state.messages.append({"role": "assistant", "content": bot_reply})
    st.chat_message("assistant").write(bot_reply, unsafe_allow_html=True)