import pandas as pd

# from TweetTemizleme.Stopwords import *
# from TweetTemizleme.main import *

# dropNaFrom(original_tweets)
# saveCsv(original_tweets,originalTweetsPath)

originalTweetsPath = 'Files/ContentOfTweets.csv'
original_tweets = pd.read_csv(originalTweetsPath, sep=",", skipinitialspace=True)  # data frame oldu

def createFrequencyTable():

    freq_tweets = original_tweets['clean_text'].apply(lambda x: pd.value_counts(x.split(" "))).sum(axis = 0).reset_index()
    freq_tweets.columns = ['words', 'frequencies']
    freq_tweets.to_csv('deneme.csv', index=None)
