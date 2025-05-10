import string
import re
import pickle
import gzip
import nltk
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import tweepy
from sklearn.feature_extraction.text import TfidfVectorizer
from scripts.youtube_utils import extract_video_id, get_youtube_comments

# ✅ Download necessary NLTK resources
nltk.download('punkt')

# ✅ Initialize Flask App
app = Flask(__name__, template_folder="templates")
CORS(app)

# ✅ Load Pre-trained Models & Vectorizer (Compressed versions)
try:
    with gzip.open('logistic_model.pkl.gz', 'rb') as f:
        logistic_model = pickle.load(f)

    with gzip.open('tree_model.pkl.gz', 'rb') as f:
        tree_model = pickle.load(f)

    with gzip.open('vectorizer.pkl.gz', 'rb') as f:
        vectorizer = pickle.load(f)

    # Load scaler if it exists
    try:
        with gzip.open('scaler.pkl.gz', 'rb') as f:
            scaler = pickle.load(f)
    except FileNotFoundError:
        scaler = None

except Exception as e:
    print("❌ Error loading compressed models/vectorizer:", str(e))
    exit(1)  # Stop execution if models are missing

# ✅ Twitter API Credentials
TWITTER_BEARER_TOKEN = "AAAAAAAAAAAAAAAAAAAAAMm2zwEAAAAAKsEg8WoT5DBS1L1rUuILqblyTxc%3DqarwlgdXWEwbRPSLNgGNG5ZGC1clOfvojAGz4H1RpqzNNCVFDV"
client = tweepy.Client(bearer_token=TWITTER_BEARER_TOKEN)

# ✅ Text Preprocessing Function (Multilingual)
def preprocess_text(text):
    text = text.lower().strip()  # Convert to lowercase  
    text = re.sub(r"http\S+|www\S+|@\S+|#", "", text)  # Remove URLs, mentions, hashtags  
    text = text.translate(str.maketrans("", "", string.punctuation))  # Remove punctuation  
    return text  # Preserve multilingual characters

# ✅ Ensure TF-IDF is Fitted
def extract_features(texts):
    processed_texts = [preprocess_text(text) for text in texts]

    # **Check if vectorizer is fitted**
    if not hasattr(vectorizer, 'vocabulary_'):
        raise ValueError("The TF-IDF vectorizer is not fitted. Train it before using.")

    return vectorizer.transform(processed_texts)

# ✅ Fetch Tweet from Link
def fetch_tweet_by_link(tweet_link):
    try:
        tweet_id = tweet_link.split("/")[-1]
        tweet = client.get_tweet(id=tweet_id, tweet_fields=["text"])
        return tweet.data.text if tweet.data else None
    except tweepy.TweepyException as e:
        return None  # Return None if error occurs (e.g., private tweet, invalid link)

# ✅ Home Route
@app.route('/')
def home():
    return render_template('index.html')

# ✅ API to Analyze Custom Text (Hindi & Multilingual Support)
@app.route('/analyze_text', methods=['POST'])
def analyze_text():
    try:
        data = request.json
        custom_text = data.get('text', '').strip()
        language = data.get('language', 'auto')  # Auto-detect if not provided

        if not custom_text:
            return jsonify({"error": "No text provided"}), 400

        # ✅ Feature Extraction
        X_tfidf = extract_features([custom_text])

        # ✅ Predictions
        logistic_pred = logistic_model.predict(X_tfidf)[0]
        tree_pred = tree_model.predict(X_tfidf)[0]

        return jsonify({
            "text": custom_text,
            "language": language,
            "logistic_prediction": "Sarcastic" if logistic_pred == 1 else "Not Sarcastic",
            "tree_prediction": "Sarcastic" if tree_pred == 1 else "Not Sarcastic"
        })

    except Exception as e:
        print("❌ Error in analyze_text:", str(e))
        return jsonify({"error": "Processing failed: " + str(e)}), 500

# ✅ API to Analyze Twitter or Custom Text
@app.route('/analyze_twitter', methods=['POST'])
def analyze_twitter():
    try:
        data = request.json
        tweet_link = data.get('tweet_link', '').strip()
        custom_tweet = data.get('tweet', '').strip()

        tweets = []
        if tweet_link:
            tweet = fetch_tweet_by_link(tweet_link)
            if not tweet:
                return jsonify({"error": "Invalid tweet link or tweet not found"}), 400
            tweets = [tweet]
        elif custom_tweet:
            tweets = [custom_tweet]
        else:
            return jsonify({"error": "Provide either a Twitter tweet link or custom text"}), 400

        # ✅ Extract Features & Predict
        X_tfidf = extract_features(tweets)
        logistic_preds = logistic_model.predict(X_tfidf).tolist()
        tree_preds = tree_model.predict(X_tfidf).tolist()

        response = []
        for i, tweet in enumerate(tweets):
            response.append({
                "text": tweet,
                "logistic_prediction": "Sarcastic" if logistic_preds[i] == 1 else "Not Sarcastic",
                "tree_prediction": "Sarcastic" if tree_preds[i] == 1 else "Not Sarcastic"
            })

        return jsonify(response)

    except Exception as e:
        print("❌ Error in analyze_twitter:", str(e))
        return jsonify({"error": "Processing failed: " + str(e)}), 500

# ✅ API to Analyze YouTube Comments
@app.route("/analyze_youtube", methods=["POST"])
def analyze_youtube():
    try:
        data = request.json
        video_url = data.get("video_url")

        # Extract video ID
        video_id = extract_video_id(video_url)
        if not video_id:
            return jsonify({"error": "Invalid YouTube URL"}), 400

        # Fetch comments
        comments = get_youtube_comments(video_id)
        if not comments:
            return jsonify({"error": "No comments found"}), 400

        # Analyze each comment
        results = []
        for comment in comments:
            X_tfidf = extract_features([comment])
            logistic_pred = logistic_model.predict(X_tfidf)[0]
            tree_pred = tree_model.predict(X_tfidf)[0]

            results.append({
                "comment": comment,
                "logistic_prediction": "Sarcastic" if logistic_pred == 1 else "Not Sarcastic",
                "tree_prediction": "Sarcastic" if tree_pred == 1 else "Not Sarcastic"
            })

        return jsonify(results)

    except Exception as e:
        print("❌ Error in analyze_youtube:", str(e))
        return jsonify({"error": "Processing failed: " + str(e)}), 500

# ✅ Run Flask App
if __name__ == '__main__':
    app.run(debug=True)
