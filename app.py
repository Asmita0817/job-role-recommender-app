import streamlit as st
import joblib

# Load trained components
model = joblib.load('job_recommender_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')
label_encoder = joblib.load('label_encoder.pkl')

st.set_page_config(page_title="ML Job Recommender", layout="centered")
st.title("💼 ML Job Role Recommender")
st.markdown("Enter your job description or skills to get a predicted job title!")

# Input from user
user_input = st.text_area("📝 Your Skills / Job Description", placeholder="e.g. Python, SQL, Data Analysis...")

# Button
if st.button("🎯 Recommend Job Role"):
    if user_input.strip() == "":
        st.warning("Please enter something first.")
    else:
        # Vectorize and predict
        X_input = vectorizer.transform([user_input])
        prediction = model.predict(X_input)
        job_title = label_encoder.inverse_transform(prediction)

        st.success(f"✅ Recommended Job Role: **{job_title[0]}**")
