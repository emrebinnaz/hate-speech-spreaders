import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


originalTweetsPath = '../Files/ContentOfTweets.csv'
wordFrequenciesPath = '../Files/WordFrequencies.csv'
tfidfPath = '../Files/tfidf.csv'
wordFrequenciesBiggerThanFivePath = '../Files/WordFrequenciesBiggerThanFive.csv'

original_tweets = pd.read_csv(originalTweetsPath, sep=",", skipinitialspace=True)

def createFrequencyTable():

    freq_tweets = original_tweets['clean_text'].apply(lambda x: pd.value_counts(x.split(" "))).sum(axis = 0).reset_index()
    freq_tweets.columns = ['words', 'frequencies']
    freq_tweets.to_csv(wordFrequenciesPath, index=None)

def filterFrequencies():

    word_frequencies = pd.read_csv(wordFrequenciesPath, sep=",", skipinitialspace=True)
    filtered_frequencies = word_frequencies[word_frequencies['frequencies'] >= 5]
    filtered_frequencies.to_csv(wordFrequenciesBiggerThanFivePath, index=None)

def createTFIDFMatrix():

    cleanTexts = original_tweets['text']
    vectorizer = TfidfVectorizer(analyzer = 'word', lowercase = True, min_df = 10)

    trainVector = vectorizer.fit_transform(cleanTexts)
    tfidf = pd.DataFrame(trainVector.toarray(), columns = vectorizer.get_feature_names())
    tfidf['label'] = original_tweets['label']
    tfidf.to_csv(tfidfPath, index=None)

# filterFrequencies() bag of words
createTFIDFMatrix()

tfidf = pd.read_csv(tfidfPath, sep=",", skipinitialspace=True)
print(tfidf.info())
