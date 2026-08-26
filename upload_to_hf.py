"""
One-time script: uploads your fine-tuned DistilBERT sentiment model to
Hugging Face Hub. Run this from your terminal (not Colab), after:
  pip install huggingface_hub
  huggingface-cli login
"""
from huggingface_hub import HfApi, create_repo

REPO_ID = "yourusername/tripadvisor-sentiment-distilbert"  # <-- change this
LOCAL_MODEL_DIR = "sentiment_model"


def main():
    api = HfApi()
    create_repo(REPO_ID, repo_type="model", exist_ok=True)
    print(f"Uploading {LOCAL_MODEL_DIR} -> {REPO_ID} ...")
    api.upload_folder(folder_path=LOCAL_MODEL_DIR, repo_id=REPO_ID, repo_type="model")
    print("Done. Your model is live at:")
    print(f"https://huggingface.co/{REPO_ID}")


if __name__ == "__main__":
    main()
