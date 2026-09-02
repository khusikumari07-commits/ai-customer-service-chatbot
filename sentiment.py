from transformers import pipeline


sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest"
)


def analyze_sentiment(text):
    result = sentiment_analyzer(text)[0]

    label = result["label"].lower()
    score = result["score"]

    if "positive" in label:
        sentiment = "positive"
    elif "negative" in label:
        sentiment = "negative"
    else:
        sentiment = "neutral"

    return sentiment, score