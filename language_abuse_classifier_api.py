from flask import Flask, request, jsonify
from transformers import BertTokenizer, BertForSequenceClassification
from langdetect import detect
import unicodedata
import torch
from flask_cors import CORS
app = Flask(__name__)
CORS(app)


# ✅ Load your fine-tuned mBERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained("./mbert_code_mixed_model")
model = BertForSequenceClassification.from_pretrained("./mbert_code_mixed_model")
model.eval()  # put the model in evaluation mode

# ✅ Script detection
def get_script(word):
    for char in word:
        if 'DEVANAGARI' in unicodedata.name(char, ''):
            return 'Devanagari'
    return 'Latin'

# ✅ Language and script detection per word
def detect_language_with_script(text):
    words = text.split()
    tags = []
    for word in words:
        try:
            lang = detect(word)
        except:
            lang = 'un'
        script = get_script(word)
        tags.append([word, lang, script])
    return tags

# ✅ Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "No text provided."}), 400

    # Tokenize and predict
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).item()

    classification = "Abusive" if prediction == 1 else "Non-abusive"
    language_tags = detect_language_with_script(text)

    return jsonify({
        "text": text,
        "language_tags": language_tags,
        "classification": classification
    })

# ✅ Start the server
if __name__ == '__main__':
    app.run(debug=True)
