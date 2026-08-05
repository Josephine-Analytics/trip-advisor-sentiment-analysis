import streamlit as st
import requests

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(
    page_title="Trip Advisor Sentiment Analysis",
    page_icon="🌍",
    layout="centered"
)

# ---------------------------
# HEADER
# ---------------------------
st.title("🌟 Trip Advisor Sentiment Analysis")
st.write("Analyze customer reviews and instantly predict sentiment ratings.")

# ---------------------------
# SIDEBAR
# ---------------------------
st.sidebar.header("About the App")
st.sidebar.write(
    """
    This app uses a fine‑tuned DistilBERT model to classify Trip Advisor reviews
    into 5 sentiment categories ranging from Very Negative to Very Positive.
    """
)

# ---------------------------
# INPUT BOX
# ---------------------------
review = st.text_area(
    "Enter a review:",
    placeholder="Type or paste a Trip Advisor review here...",
    height=150
)

# ---------------------------
# PREDICT BUTTON
# ---------------------------
if st.button("🔍 Predict Sentiment"):

    if review.strip():

        # Call Flask API
        response = requests.post(
            "http://127.0.0.1:5000/predict",
            json={"review": review}
        )
        data = response.json()

        # Extract values returned by API
        label = data["label"]
        predicted_class = data["prediction"]
        confidence = data["confidence"]   # REAL confidence from API

        # ---------------------------
        # EMOJIS
        # ---------------------------
        emoji_map = {
            "Rating 5 (Very Positive)": "😍",
            "Rating 4 (Positive)": "🙂",
            "Rating 3 (Neutral)": "😐",
            "Rating 2 (Negative)": "🙁",
            "Rating 1 (Very Negative)": "😡"
        }
        emoji = emoji_map[label]

        # ---------------------------
        # COLORS
        # ---------------------------
        color_map = {
            "Rating 5 (Very Positive)": "#d4edda",
            "Rating 4 (Positive)": "#e2f0cb",
            "Rating 3 (Neutral)": "#fff3cd",
            "Rating 2 (Negative)": "#f8d7da",
            "Rating 1 (Very Negative)": "#f5c6cb"
        }
        bg_color = color_map[label]

        # ---------------------------
        # RESULT CARD
        # ---------------------------
        st.markdown(
            f"""
            <div style="padding: 20px; border-radius: 10px; background-color: {bg_color};">
                <h3 style="margin: 0;">{emoji} {label}</h3>
                <p style="margin: 0;">Model Class: {predicted_class}</p>
                <p style="margin: 0;"><b>Confidence:</b> {confidence:.2f}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    else:
        st.warning("Please enter a review first.")

# ---------------------------
# FOOTER
# ---------------------------
st.markdown(
    """
    <hr>
    <p style='text-align:center; color:#888;'>
    Built by <b>Josephine Namyalo</b> • Powered by DistilBERT 🌍
    </p>
    """,
    unsafe_allow_html=True
)
