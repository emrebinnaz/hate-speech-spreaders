from textblob import TextBlob

def getSentimentalPolarity(text):

    polarity = TextBlob(text).sentiment.polarity
    print(polarity)

    if polarity < 0:
        return "negatif"
    elif polarity == 0:
        return "nötr"
    elif polarity > 0:
        return "pozitif"
