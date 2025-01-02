import streamlit as st
import pickle
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize

nltk.download('punkt_tab')
nltk.download('stopwords')

# Initialize stemmer
ps = PorterStemmer()

# Predefined regex patterns
URL_PATTERN = r"https?://\S+|www\.\S+"
PUNCT_DIGIT_PATTERN = r"[^a-zA-Z\s]"

# Text Preprocessing Functions
def remove_url(text):
    return re.sub(URL_PATTERN, '', text)

def remove_punct_and_digits(text):
    return re.sub(PUNCT_DIGIT_PATTERN, '', text)

def remove_stopwords(words):
    stop_words = set(stopwords.words('english'))
    return [word for word in words if word not in stop_words]

def stem_words(words):
    return [ps.stem(word) for word in words]

def transform_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove URLs
    text = remove_url(text)
    # Remove punctuation and digits
    text = remove_punct_and_digits(text)
    # Tokenize
    words = word_tokenize(text)
    # Remove stopwords
    words = remove_stopwords(words)
    # Stem words
    words = stem_words(words)
    # Rejoin to a single string
    return " ".join(words)

# Load models and vectorizer
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Streamlit Application
st.title("SMS Spam Classifier")

# Input from user
input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    # 1. Preprocess the input
    transformed_sms = transform_text(input_sms)
    # 2. Vectorize the input
    vector_input = tfidf.transform([transformed_sms])
    # 3. Predict the result
    result = model.predict(vector_input)[0]
    # 4. Display the result
    st.header("Spam" if result == 1 else "Not Spam")
