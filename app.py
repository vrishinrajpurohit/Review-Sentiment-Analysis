from flask import Flask, request, render_template
import tensorflow as tf
import pickle
import numpy as np
import re

app = Flask(__name__)


model = tf.keras.models.load_model('sentiment_model/model.keras')
with open('sentiment_model/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

def clean_text(text):

    text = re.sub(r'[^a-zA-Z0-9!?]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

def preprocess_text(text, max_length=200):

    text = clean_text(text)
    sequences = tokenizer.texts_to_sequences([text])
    padded_seq = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_length, padding='post')
    return padded_seq

def predict_sentiment(text):
    processed_text = preprocess_text(text)
    prediction = model.predict(processed_text)

    sentiment = "Positive" if prediction[0][0] >= 0.5 else "Negative"
    probability = prediction[0][0] if sentiment == "Positive" else 1 - prediction[0][0]
    return sentiment, round(probability * 100, 2)

@app.route('/', methods=['GET', 'POST'])
def index():
    sentiment = None
    confidence = None

    if request.method == 'POST':
        review = request.form['review']
        if review.strip():
            sentiment, confidence = predict_sentiment(review)

    return render_template('index.html', sentiment=sentiment, confidence=confidence)

if __name__ == '__main__':
    app.run(debug=True)
