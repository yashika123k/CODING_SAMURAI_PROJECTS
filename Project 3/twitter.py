import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from ntscraper import Nitter

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="🌸",
    layout="centered"
)

# -----------------------------
# Pastel + kawaii CSS
# -----------------------------
st.markdown("""
<style>
/* App background */
.stApp {
    background: linear-gradient(135deg, #fdf2f8, #ecfeff);
    font-family: "Segoe UI", sans-serif;
}

/* Title */
h1 {
    text-align: center;
    color: #9333ea;
}

/* Subtitle */
.caption {
    text-align: center;
    color: #6b7280;
}

/* Inputs */
textarea, input {
    border-radius: 16px !important;
    border: 2px solid #fbcfe8 !important;
}

/* Selectbox */
div[data-baseweb="select"] {
    border-radius: 16px;
}

/* Animated buttons */
.stButton > button {
    background: linear-gradient(135deg, #f9a8d4, #c4b5fd);
    color: white;
    border-radius: 999px;
    border: none;
    padding: 0.6em 1.6em;
    font-size: 16px;
    transition: all 0.3s ease;
}
.stButton > button:hover {
    transform: scale(1.08);
    background: linear-gradient(135deg, #f472b6, #a78bfa);
}

/* Cards */
.kawaii-card {
    padding: 16px;
    border-radius: 20px;
    margin-bottom: 14px;
    box-shadow: 0 10px 20px rgba(0,0,0,0.08);
    animation: pop 0.4s ease;
}

/* Card colors */
.positive {
    background-color: #bbf7d0;
}
.negative {
    background-color: #fecaca;
}

/* Card text */
.kawaii-card p {
    color: #1f2937;
    font-size: 16px;
}
.kawaii-card h4 {
    color: #111827;
}

/* Sentiment text */
.sentiment-positive {
    color: #16a34a; /* pastel green */
    font-size: 20px;
    font-weight: bold;
}
.sentiment-negative {
    color: #dc2626; /* pastel red */
    font-size: 20px;
    font-weight: bold;
}

/* Card pop animation */
@keyframes pop {
    from { transform: scale(0.95); opacity: 0; }
    to { transform: scale(1); opacity: 1; }
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Load stopwords
# -----------------------------
@st.cache_resource
def load_stopwords():
    nltk.download("stopwords")
    return set(stopwords.words("english"))

# -----------------------------
# Load model & vectorizer
# -----------------------------
@st.cache_resource
def load_model_and_vectorizer():
    with open("twitter.pkl", "rb") as model_file:
        model = pickle.load(model_file)
    with open("vectorizer.pkl", "rb") as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    return model, vectorizer

# -----------------------------
# Text preprocessing
# -----------------------------
CLEAN_RE = re.compile(r"[^a-zA-Z]")

def preprocess(text, stop_words):
    words = CLEAN_RE.sub(" ", text).lower().split()
    return " ".join(word for word in words if word not in stop_words)

# -----------------------------
# Sentiment prediction
# -----------------------------
def predict_sentiment(text, model, vectorizer, stop_words):
    processed = preprocess(text, stop_words)
    vectorized = vectorizer.transform([processed])
    prediction = model.predict(vectorized)[0]
    return "Positive" if prediction == 1 else "Negative"

# -----------------------------
# Initialize Nitter
# -----------------------------
@st.cache_resource
def initialize_scraper():
    return Nitter(log_level=1)

# -----------------------------
# Kawaii card
# -----------------------------
def create_card(tweet_text, sentiment):
    emoji = "💖" if sentiment == "Positive" else "💔"
    card_class = "positive" if sentiment == "Positive" else "negative"
    sentiment_class = "sentiment-positive" if sentiment == "Positive" else "sentiment-negative"

    return f"""
    <div class="kawaii-card {card_class}">
        <h4>{emoji} <span class="{sentiment_class}">{sentiment} Sentiment</span></h4>
        <p>{tweet_text}</p>
    </div>
    """

# -----------------------------
# Main app
# -----------------------------
def main():
    st.title("🌸Twitter Sentiment Analyzer 🌸")
    st.markdown(
        "<p class='caption'>✨ cute vibes • smart AI • real tweets ✨</p>",
        unsafe_allow_html=True
    )

    stop_words = load_stopwords()
    model, vectorizer = load_model_and_vectorizer()
    scraper = initialize_scraper()

    option = st.selectbox(
        "🎀 Choose a mode",
        ["💬 Analyze text", "🐦 Analyze Twitter user"]
    )

    if option == "💬 Analyze text":
        text_input = st.text_area("✍️ Type something cute (or spicy)")

        if st.button("✨ Analyze Sentiment ✨"):
            sentiment = predict_sentiment(
                text_input, model, vectorizer, stop_words
            )
            emoji = "🥰" if sentiment == "Positive" else "🥺"
            css_class = "sentiment-positive" if sentiment == "Positive" else "sentiment-negative"
            st.markdown(
                f"<p class='{css_class}'>{emoji} Sentiment: {sentiment}</p>",
                unsafe_allow_html=True
            )

    else:
        username = st.text_input("🐤 Twitter username (without @)")

        if st.button("🍡 Fetch Tweets 🍡"):
            tweets_data = scraper.get_tweets(
                username, mode="user", number=5
            )

            if "tweets" in tweets_data:
                for tweet in tweets_data["tweets"]:
                    sentiment = predict_sentiment(
                        tweet["text"], model, vectorizer, stop_words
                    )
                    st.markdown(
                        create_card(tweet["text"], sentiment),
                        unsafe_allow_html=True
                    )
            else:
                st.warning("🥺 No tweets found")

# -----------------------------
# Run app
# -----------------------------
if __name__ == "__main__":
    main()
