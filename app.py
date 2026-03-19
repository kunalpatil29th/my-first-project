import streamlit as st
import pickle
from streamlit_lottie import st_lottie
import requests

# Function to load Lottie animation from URL
def load_lottie(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Load Lottie animation
lottie_animation = load_lottie("https://assets6.lottiefiles.com/packages/lf20_touohxv0.json")

# Display animation at the top
st_lottie(lottie_animation, height=300, key="background")

# Load ML model & vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# App title
st.title("📰 Fake News Detector ")
st.write("Enter a news article to check whether it is REAL or FAKE")

# Text input
news_text = st.text_area("News Text")

# Button
if st.button("Check News"):
    if news_text.strip() == "":
        st.warning("Please enter some text")
    else:
        vec_text = vectorizer.transform([news_text])
        prediction = model.predict(vec_text)

        if prediction[0] == "FAKE":
            st.error("❌ This news is FAKE")
        else:
            st.success("✅ This news is REAL")
