import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('punkt_tab')


model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    tokens = nltk.word_tokenize(text.lower())
    tokens = [t for t in tokens if t.isalpha()]
    tokens = [t for t in tokens if t not in stop_words]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)

st.set_page_config(page_title="📰 Fake News Detector", layout="centered")
st.title("📰 Fake News Detection App")
st.write("Enter a news article or headline to check if it's **Fake** or **Real**.")

user_input = st.text_area("✍️ Paste News Content Here")

if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        cleaned = preprocess(user_input)
        vec = vectorizer.transform([cleaned])
        prediction_probs = model.predict_proba(vec)[0]

        predicted_class = model.classes_[prediction_probs.argmax()]

        if predicted_class == 1: 
            label = "✅ Real News"
        else:
            label = "🚫 Fake News"

        st.subheader(f"Prediction: {label}")
        st.write(f"Confidence scores: Fake = {prediction_probs[0]:.2f}, Real = {prediction_probs[1]:.2f}")
