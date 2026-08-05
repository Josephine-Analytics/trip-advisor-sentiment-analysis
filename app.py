from flask import Flask, request, jsonify
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

app = Flask(__name__)

# Load model
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertForSequenceClassification.from_pretrained("sentiment_model")
model.eval()

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    review = data["review"]

    # Tokenize
    inputs = tokenizer(review, return_tensors="pt", truncation=True, padding=True)

    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Softmax probabilities
    probs = torch.softmax(logits, dim=1)
    predicted_class = torch.argmax(probs, dim=1).item()
    confidence = probs[0][predicted_class].item()

    # Map class → label
    label_map = {
        0: "Rating 1 (Very Negative)",
        1: "Rating 2 (Negative)",
        2: "Rating 3 (Neutral)",
        3: "Rating 4 (Positive)",
        4: "Rating 5 (Very Positive)"
    }
    label = label_map[predicted_class]

    return jsonify({
        "label": label,
        "prediction": predicted_class,
        "confidence": confidence
    })

if __name__ == "__main__":
    app.run(debug=True)

