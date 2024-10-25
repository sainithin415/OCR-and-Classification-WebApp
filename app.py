from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pytesseract
import numpy as np
import torch
from PIL import Image
import re

app = Flask(__name__)
model = load_model('models/model.h5')  # Load pre-trained model
tokenizer = AutoTokenizer.from_pretrained('t5-base')
summarizer = AutoModelForSeq2SeqLM.from_pretrained('t5-base')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    img = Image.open(file).convert('L').resize((28, 28))
    img_array = np.array(img).reshape(1, 28, 28, 1) / 255.0
    prediction = model.predict(img_array)
    label = np.argmax(prediction, axis=1)
    return jsonify({'prediction': int(label[0])})

@app.route('/extract-text', methods=['POST'])
def extract_text():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    text = pytesseract.image_to_string(file, lang='eng')
    inputs = tokenizer.encode("summarize: " + text, return_tensors='pt', truncation=True)
    summary_ids = summarizer.generate(inputs, max_length=50, min_length=10, length_penalty=4.0, num_beams=2)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return jsonify({'text': text, 'summary': summary})

if __name__ == '__main__':
    app.run(debug=True)
