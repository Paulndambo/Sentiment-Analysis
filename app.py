import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the model and tokenizer
@st.cache_resource
def load_model_and_tokenizer():
    model = tf.keras.models.load_model("model.h5")
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()

# App title
st.title("🔍 Sentiment Analysis App")
st.write("Enter a sentence below to analyze its sentiment:")

# Input field
user_input = st.text_area("Your Text", height=150)

# Prediction function
def predict_sentiment(text):
    max_length = 50  # Same maxlen used during training
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_length, padding='post', truncating='post')
    prediction = model.predict(padded)
    sentiment_label = np.argmax(prediction)
    confidence = float(np.max(prediction))

    if sentiment_label == 0:
        sentiment = "Negative 😠"
    elif sentiment_label == 1:
        sentiment = "Neutral 😐"
    else:
        sentiment = "Positive 😊"

    return sentiment, confidence

# Predict button
if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        sentiment, confidence = predict_sentiment(user_input)
        st.success(f"**Sentiment:** {sentiment}")
        st.info(f"**Confidence:** {confidence:.2f}")
