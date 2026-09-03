import os
import re

import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

# The fine-tuned model on Hugging Face Hub: huggingface.co/JosephineNamyalo/Sentiment_Model
# Can be overridden via the HF_MODEL_REPO environment variable (set this on Render).
MODEL_REPO = os.environ.get("HF_MODEL_REPO", "JosephineNamyalo/Sentiment_Model")

# LabelEncoder sorts classes alphabetically at training time:
# negative -> 0, neutral -> 1, positive -> 2
LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}

app = Flask(__name__)
CORS(app)  # allows a browser-based frontend (hosted anywhere) to call this API

print(f"Loading model from Hugging Face Hub: {MODEL_REPO} ...")
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_REPO)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_REPO)
model.eval()  # inference mode: disables dropout etc.
print("Model loaded.")


def clean_text(text):
    text = str(text)
    text = re.sub(r"<.*?>", " ", text)               # remove HTML
    text = re.sub(r"http\S+|www\.\S+", " ", text)     # remove URLs
    text = text.lower()
    text = text.encode("ascii", "ignore").decode()    # strip emojis/non-ascii
    text = re.sub(r"\s+", " ", text).strip()
    return text


@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "tripadvisor-sentiment-transformer-api"})


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)
    if not data or "review" not in data:
        return jsonify({"error": "Send JSON like {\"review\": \"your text here\"}"}), 400

    review = data["review"]
    if not isinstance(review, str) or not review.strip():
        return jsonify({"error": "'review' must be a non-empty string"}), 400

    cleaned = clean_text(review)

    inputs = tokenizer(
        cleaned,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256,
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)[0]

    pred_idx = int(torch.argmax(probs).item())

    response = {
        "sentiment": LABEL_MAP[pred_idx],
        "confidence": {
            LABEL_MAP[i]: round(float(p), 4) for i, p in enumerate(probs.tolist())
        },
    }
    return jsonify(response)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)  # debug=False for production
