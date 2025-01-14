import json
import nltk
import numpy as np
import streamlit as st
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

# Download necessary NLTK data files
nltk.download("punkt")
nltk.download("wordnet")

# Load the intents JSON file
with open("intents.json") as file:
    intents = json.load(file)

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Preprocess the data: prepare patterns, tags, and responses
patterns = []
tags = []
responses = {}

for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        patterns.append(pattern)
        tags.append(intent["tag"])
    responses[intent["tag"]] = intent["responses"]

# Tokenize and lemmatize the words
def clean_text(text):
    return [lemmatizer.lemmatize(word.lower()) for word in nltk.word_tokenize(text)]

# Train the model using a pipeline
vectorizer = TfidfVectorizer(tokenizer=clean_text)
model = make_pipeline(vectorizer, LogisticRegression())

# Fit the model
model.fit(patterns, tags)

# Streamlit UI setup
st.title("Chatbot")
st.write("Ask me anything!")

# Get user input and predict the response
user_input = st.text_input("You: ", "")

if user_input:
    # Predict the tag of the user input
    tag = model.predict([user_input])[0]
    
    # Get the response associated with the tag
    response = np.random.choice(responses[tag])
    
    # Display the response
    st.write(f"Chatbot: {response}")
