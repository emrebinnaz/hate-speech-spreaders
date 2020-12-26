import pandas as pd
import nltk
import preprocessor as preprocessor
import os
from nltk.corpus import stopwords
from nltk.tokenize.toktok import ToktokTokenizer

words = set(nltk.corpus.words.words())
stopWordsList = nltk.corpus.stopwords.words('english')
tokenizer=ToktokTokenizer()


tweets = pd.read_csv('ContentOfTweetsYedek.csv', sep=",", skipinitialspace=True)  # data frame oldu

def convertDataTypes():
    tweets['id'] = tweets['id'].astype('string')
    tweets['text'] = tweets['text'].astype('string')
    tweets['label'] = tweets['label'].astype('category')

def dropNaFrom(tweets):

    tweets.dropna(how = "any", inplace = True)

def removeNumbersFrom(tweets):

    tweets.text = tweets.text.replace('\d+', '', regex = True)

def removeRtFrom(tweets):

    tweets.text = tweets.text.replace("RT", '', regex=True)

def removeSpacesFrom(tweets):

    tweets.text = tweets.text.replace('\s+', ' ', regex = True)

def addQuotesToEndOfTweetText(tweets):

    tweets.update(tweets[['text']].applymap('"{}" '.format))

def createTxtForPreprocessing():
    tweets.to_csv("ContentOfTweetsYedek.csv", sep=',', index=False)
    output = open('CleanTxtFile.txt', 'w+', errors="ignore", encoding="ISO-8859-1")
    with open('ContentOfTweetsYedek.csv', "rt", encoding="ISO-8859-1") as f:
        for row in f:
            output.write(row)

def applyPreprocessing(tweets):

    dropNaFrom(tweets)
    removeNumbersFrom(tweets)
    removeRtFrom(tweets)
    removeSpacesFrom(tweets)
    addQuotesToEndOfTweetText(tweets)
    createTxtForPreprocessing()

    preprocessor.set_options(preprocessor.OPT.MENTION,
                             preprocessor.OPT.URL,
                             preprocessor.OPT.RESERVED,
                             preprocessor.OPT.EMOJI,
                             preprocessor.OPT.SMILEY)

    return preprocessor.clean_file("CleanTxtFile.txt")

def createCleanCsvFrom(cleanTxtFile):
    readFile = pd.read_csv(cleanTxtFile, sep=",")
    readFile.to_csv('ContentOfTweetsYedek.csv', index=None)
    os.remove(cleanTxtFile)

def removePunctuations(): # daha sonra hashtagler için # sembolü geri getirilmeli

    tweets = pd.read_csv('ContentOfTweetsYedek.csv', sep=",", skipinitialspace=True)  # data frame oldu
    tweets.text = tweets.text.str.replace('[^\w\s]', '')
    tweets.to_csv('ContentOfTweetsYedek.csv', index=None)

def readCleanFile():
    tweets = pd.read_csv('ContentOfTweetsYedek.csv', sep=",")  # data frame oldu
    convertDataTypes()
    print(tweets.info())

def cleanNAandSpaceFromOriginalFile():

    original_tweets = pd.read_csv('ContentOfTweets.csv', sep=",", skipinitialspace=True)  # data frame oldu
    dropNaFrom(original_tweets)
    removeSpacesFrom(original_tweets)
    original_tweets.to_csv('ContentOfTweets.csv', index=None)

def addCleanTextToOriginalFile():

    tweets = pd.read_csv('ContentOfTweetsYedek.csv', sep=",", skipinitialspace=True)  # data frame oldu
    original_tweets = pd.read_csv('ContentOfTweets.csv', sep=",", skipinitialspace=True)  # data frame oldu
    original_tweets['clean_text'] = tweets['text']
    original_tweets['label'] = tweets['label']
    original_tweets.to_csv('ContentOfTweets.csv', index=None)
    print(original_tweets.info())

def removeNonEnglishWordsFrom(tweets):

    tweets['text'] = tweets['text'].apply(lambda x:  " ".join(w for w in nltk.wordpunct_tokenize(x) \
         if w.lower() in words or not w.isalpha()))
    tweets.to_csv('ContentOfTweetsYedek.csv', index=None)

def makeLowercaseTo(tweets):

    tweets['clean_text'] = tweets['clean_text'].apply(lambda text: " ".join(text.lower() for text in text.split()))

def removeStopwordsFrom(text):
    # set stopwords to english

    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]

    filtered_tokens = [token for token in tokens if token not in stopWordsList]

    filtered_text = ' '.join(filtered_tokens)
    return filtered_text




# Main commands
# cleanTxtFile = applyPreprocessing(tweets)
# createCleanCsvFrom(cleanTxtFile)
# removePunctuations()
# addCleanTextToOriginalFile()
# readCleanFile()

original_tweets = pd.read_csv('ContentOfTweets.csv', sep=",", skipinitialspace=True)  # data frame oldu
#makeLowercaseTo(original_tweets)
#original_tweets['clean_text'] = original_tweets['clean_text'].apply(removeStopwordsFrom)

original_tweets.to_csv('ContentOfTweets.csv', index=None)
print(original_tweets.info())
