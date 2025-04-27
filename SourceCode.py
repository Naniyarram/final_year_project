import tweepy
import re
from django.shortcuts import render
import wikipediaapi
import requests
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Twitter API setup
BEARER_TOKEN = "AAAAAAAAAAAAAAAAAAAAAGd40wEAAAAADiTarmcnJhfnaeUvPflvPZgOdvY%3D6g5mo3NQ3beb4xpZahMEgZdYWWJ4u25LzyVxiqCnM4T14chuVz"  # Replace with your friend's token
client = tweepy.Client(bearer_token=BEARER_TOKEN)

# Wikipedia API setup
wiki_wiki = wikipediaapi.Wikipedia(user_agent='TweetSentimentAnalyzer (yarra@example.com)', language='en')

# Load the LSTM model and tokenizer
lstm_model = load_model('lstm_sentiment_model.h5')
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

# LSTM preprocessing parameters (must match training script)
max_len = 50

def preprocess_tweet_for_lstm(tweet_text):
    # Basic cleaning (similar to training)
    tweet_text = tweet_text.lower()
    tweet_text = re.sub(r"http\S+|www\S+|https\S+", "", tweet_text, flags=re.MULTILINE)
    tweet_text = re.sub(r"@\w+|#\w+", "", tweet_text)
    # Keep emojis for LSTM to learn from them
    return tweet_text

def get_sentiment_lstm(tweet_text):
    # Preprocess the tweet
    cleaned_text = preprocess_tweet_for_lstm(tweet_text)
    # Tokenize and pad the text
    sequence = tokenizer.texts_to_sequences([cleaned_text])
    padded = pad_sequences(sequence, maxlen=max_len, padding='post', truncating='post')
    # Predict sentiment
    prediction = lstm_model.predict(padded, verbose=0)
    sentiment_idx = np.argmax(prediction, axis=1)[0]
    # Map index to sentiment
    if sentiment_idx == 2:
        return "Positive 😊"
    elif sentiment_idx == 0:
        return "Negative 😞"
    else:
        return "Neutral 😐"

def fetch_tweets(query, count=20):
    try:
        # Add -is:retweet to the query to exclude retweets
        modified_query = f"{query} -is:retweet"
        tweets = client.search_recent_tweets(query=modified_query,
                                             tweet_fields=["created_at", "author_id"],
                                             max_results=count,
                                             expansions=["author_id"])
        tweet_list = []
        seen_texts = set()  # To track unique tweet texts
        
        if tweets.data:
            for tweet in tweets.data:
                # Skip if we've already seen this tweet text
                if tweet.text in seen_texts:
                    continue
                seen_texts.add(tweet.text)
                
                sentiment = get_sentiment_lstm(tweet.text)  # Use LSTM for sentiment
                tweet_list.append({
                    "original_text": tweet.text,
                    "sentiment": sentiment,
                    "author_id": tweet.author_id,
                    "created_at": str(tweet.created_at)
                })
        return tweet_list
    except tweepy.TweepyException as e:
        if e.response and e.response.status_code == 429:
            return {"error": "Rate limit hit, please try again in 15 minutes!"}
        return {"error": f"Twitter API error: {str(e)}"}
    except requests.exceptions.ConnectionError:
        return {"error": "Oops! Couldn’t connect to Twitter—check your internet or try again later!"}

def fetch_wikipedia_summary(query):
    try:
        # Clean the query by removing hashtags and spaces
        cleaned_query = query.replace("#", "").strip()
        # Try different variations of the query
        possible_titles = [
            cleaned_query,  # e.g., #pushpa2 → pushpa2
            cleaned_query.replace("pushpa2", "Pushpa 2"),  # e.g., Pushpa 2
            cleaned_query.replace("pushpa2", "Pushpa 2: The Rule"),  # e.g., Pushpa 2: The Rule
            "Pushpa 2: The Rule",  # Direct match
            "Pushpa: The Rule"  # Alternative
        ]
        
        for title in possible_titles:
            page = wiki_wiki.page(title)
            if page.exists():
                # Get the summary (first paragraph) and limit to 2-3 sentences
                summary = page.summary.split('. ')[:2]  # First 2 sentences
                return '. '.join(summary) + ('.' if summary else '')
        
        return f"No Wikipedia page found for '{cleaned_query}'. Try searching with a more specific term."
    except Exception as e:
        return f"Error fetching Wikipedia summary: {str(e)}"

def analyze_tweets(request):
    if request.method == "POST":
        query = request.POST.get("query")
        tweets = fetch_tweets(query)
        if isinstance(tweets, dict) and "error" in tweets:
            return render(request, "analyzer/search.html", {"error": tweets["error"]})
        
        # Calculate sentiment counts for pie chart
        positive = sum(1 for t in tweets if "Positive" in t["sentiment"])
        negative = sum(1 for t in tweets if "Negative" in t["sentiment"])
        neutral = sum(1 for t in tweets if "Neutral" in t["sentiment"])
        total = len(tweets)
        
        # Percentages for pie chart
        chart_data = {
            "positive": (positive / total * 100) if total > 0 else 0,
            "negative": (negative / total * 100) if total > 0 else 0,
            "neutral": (neutral / total * 100) if total > 0 else 0,
            "positive_count": positive,
            "negative_count": negative,
            "neutral_count": neutral
        }
        
        # Fetch Wikipedia summary
        wiki_summary = fetch_wikipedia_summary(query)
        # Fallback if Wikipedia summary fails
        if "No Wikipedia page found" in wiki_summary or "Error fetching" in wiki_summary:
            wiki_summary = f"Analyzing sentiments for '{query}'. This shows how people feel about this topic on Twitter."
        
        return render(request, "analyzer/results.html", {
            "tweets": tweets,
            "query": query,
            "chart_data": chart_data,
            "wiki_summary": wiki_summary
        })
    return render(request, "analyzer/search.html")
