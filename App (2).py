import streamlit as st
import pickle
import numpy as np
import sklearn

# Load model and scaler
with open('spam_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    scaler = pickle.load(f)
st.title("💌 Spam Email Classifier")
st.write("Enter the email message below to classify it as spam or ham.")
email = st.text_area("Enter email text here")
if st.button("Classify"):
    if email:
        email_vectorized = scaler.transform([email])
        prediction = model.predict(email_vectorized)
        if prediction[0] == 1:
            st.error("The email is classified as Spam.")
        else:
            st.success("The email is classified as Ham.")
