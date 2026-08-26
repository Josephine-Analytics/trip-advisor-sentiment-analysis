# TripAdvisor Sentiment API — DistilBERT Transformer

Predicts sentiment (`positive` / `neutral` / `negative`) for a hotel review using a fine-tuned DistilBERT model, served with Flask and deployed on Render. The model is hosted on Hugging Face Hub, not committed to this repo, so no Git LFS is needed.

Model: DistilBERT (base-uncased), fine-tuned on the TripAdvisor hotel reviews dataset for 3-class sentiment classification. **88% accuracy** on a held-out test set — outperforms a TF-IDF + SMOTE baseline (83%) across every class, with the biggest gains on negative and neutral sentiment.

Live model: [huggingface.co/JosephineNamyalo/Sentiment_Model](https://huggingface.co/JosephineNamyalo/Sentiment_Model)

## Project files

TripAdvisor Sentiment API — DistilBERT Transformer

Predicts sentiment (positive / neutral / negative) for a hotel review using a fine-tuned DistilBERT model, served with Flask and deployed on Render. The model is hosted on Hugging Face Hub, not committed to this repo, so no Git LFS is needed.

Model: DistilBERT (base-uncased), fine-tuned on the TripAdvisor hotel reviews dataset for 3-class sentiment classification. 88% accuracy on a held-out test set — outperforms a TF-IDF + SMOTE baseline (83%) across every class, with the biggest gains on negative and neutral sentiment.

Live model: huggingface.co/JosephineNamyalo/Sentiment_Model

Project files
app.py                Flask API (loads model from Hugging Face Hub, exposes /predict)
upload_to_hf.py         Script used to push the trained model to HF Hub (already run)
requirements.txt        Python dependencies (torch CPU build + transformers)
Procfile                 Tells Render how to start the app
render.yaml               Render Blueprint config (targets a paid tier)
Run locally
bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
python3 app.py

First run will be slower while the model downloads from Hugging Face and loads into memory. The app runs on http://localhost:5000.

Test it:

bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"review": "Great location, friendly staff, would stay again"}'

Response:

json
{
  "sentiment": "positive",
  "confidence": {"negative": 0.0004, "neutral": 0.0036, "positive": 0.996}
}
Deploy to Render

Important: this model is heavier than a classical ML model. DistilBERT + PyTorch need more RAM than Render's free tier provides (512MB is not enough). Use at least the Starter plan (check Render's current pricing — this was accurate as of when this was built, but plans and prices can change).

Push this folder to a GitHub repo (a normal git push works fine — the model isn't in this repo, so no Git LFS is needed):
bash
   git init
   git add .
   git commit -m "TripAdvisor sentiment API (transformer, HF Hub)"
   git remote add origin <your-repo-url>
   git push -u origin main
Go to https://dashboard.render.com → New → Web Service.
Connect the repo. Render will detect render.yaml automatically, or set manually:
Environment: Python 3
Build Command: pip install -r requirements.txt
Start Command: gunicorn app:app --timeout 120 --workers 1
Plan: Starter (or higher) — not Free
Confirm the HF_MODEL_REPO environment variable is set to JosephineNamyalo/Sentiment_Model (already the default in app.py and render.yaml, but double-check it on Render's dashboard too).
Click Create Web Service. On first boot, the app downloads the model from Hugging Face — expect the first deploy (and the first cold start after any sleep) to take longer than a typical Flask app.

Test the deployed version the same way as local, just swap the base URL.

If the Hugging Face model repo is private, also set an HF_TOKEN environment variable on Render (a "read" token is enough) so from_pretrained() can authenticate. If it's public, no token is needed at all.

On cold starts: even on a paid tier, downloading + loading DistilBERT takes several seconds to a couple minutes on first start. If auto-sleep is enabled on a low tier, expect a noticeably slower first request after idle periods.

API reference
GET /

Health check. Returns {"status": "ok"}.

POST /predict

Body: {"review": "<text>"}

Returns:

json
{
  "sentiment": "positive | neutral | negative",
  "confidence": {"positive": 0.0, "neutral": 0.0, "negative": 0.0}
}

Errors (400) if review is missing or empty.

Why this model over the TF-IDF baseline
Metric	TF-IDF + SMOTE	DistilBERT
Accuracy	83%	88%
Macro F1	0.69	0.75
Negative F1	0.77	0.84
Neutral F1	0.39	0.46
Positive F1	0.91	0.94

Both models were evaluated on the identical test split (random_state=42, stratify=sentiment_label), so this comparison is apples-to-apples. The transformer's gains are largest on negative and neutral sentiment — cases where word order, negation, and tone matter more than which individual words appear, which bag-of-words models like TF-IDF can't capture.
