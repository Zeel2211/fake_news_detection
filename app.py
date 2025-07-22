import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Preprocess function
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    tokens = nltk.word_tokenize(text.lower())
    tokens = [t for t in tokens if t.isalpha()]
    tokens = [t for t in tokens if t not in stop_words]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)

# Streamlit UI
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
        prediction = model.predict(vec)[0]
        label = "✅ Real News" if prediction == 1 else "🚫 Fake News"
        st.subheader(f"Prediction: {label}")

