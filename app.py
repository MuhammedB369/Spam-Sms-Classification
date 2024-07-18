import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import joblib

# Load the TFIDF vectorizer and models
tfidf = joblib.load('tfidf_vectorizer.pkl')
models = {
    "Multinomial Naive Bayes": joblib.load('MultinomialNB_model.pkl'),
    "Decision Tree": joblib.load('DecisionTree_model.pkl'),
    "Random Forest": joblib.load('RandomForest_model.pkl')
}

# Text preprocessing function
def preprocess_text(text):
    wnl = WordNetLemmatizer()
    text = re.sub(pattern='[^a-zA-Z]', repl=' ', string=text)
    text = text.lower()
    words = text.split()
    filtered_words = [word for word in words if word not in set(stopwords.words('english'))]
    lemmatized_words = [wnl.lemmatize(word) for word in filtered_words]
    return ' '.join(lemmatized_words)

# Prediction function
def predict_spam(model, message):
    message = preprocess_text(message)
    message_vec = tfidf.transform([message]).toarray()
    return model.predict(message_vec)

# Streamlit interface
st.title("SMS Spam Detection")

# Select model
model_name = st.selectbox("Choose a model", list(models.keys()))
model = models[model_name]

# Input message
sample_message = st.text_area("Enter a message to classify", "You could be entitled up to £3,160 in compensation from mis-sold PPI on a credit card or loan. Please reply PPI for info or STOP to opt out.")

if st.button("Predict"):
    prediction = predict_spam(model, sample_message)
    if prediction[0] == 1:
        st.write("Gotcha! This is a SPAM message.")
    else:
        st.write("This is a HAM (normal) message.")

