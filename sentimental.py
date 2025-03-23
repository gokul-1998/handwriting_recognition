from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# Download the model"lexicon" for analysis
# nltk.download('vader_lexicon')

# Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()

def analyze(text):
    scores = sia.polarity_scores(text)
    # classification of sntiment using the scores
    if scores['compound'] >= 0.05:
        sentiment = "Positive"
    elif scores['compound'] <= -0.05:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
    # return {"Text": text, "Analysis": sentiment, "scores": scores}
    return "Analysis: " + sentiment

#function call
line = "I love this amazing project!"
result = analyze(line)
print(result)
