import pandas as pd
import os

from nltk.stem import WordNetLemmatizer
from nltk.tokenize.toktok import ToktokTokenizer
from Stopwords import createNewStopWordList, createNewSpecialNameList
from Config.Lemmatization import *
from Config.Preprocessor import getPreprocessor

words = set(nltk.corpus.words.words())
tokenizer = ToktokTokenizer()

## ContentOfTweets silinirse çalıştırılacak file sırası:

# DatasetWithTweetIdFromTweetAccess.csv
# DatasetWithoutTweetId.csv
# DatasetWithoutTweetId2.csv
# DatasetWith27000Hateful.csv
# DatasetOfNormalTweetsWithTweetId.csv
# DatasetWith3500Normal.csv

tweetsPath = '../Files/CleanDatasets/DatasetWith3500Normal.csv' #gelen verisetine göre path değiştir !!!
cleanTweetsPath = '../Files/ContentOfTweets.csv'

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
    tweets.to_csv(tweetsPath, sep=',', index=False)
    output = open('../Files/CleanTxtFile.txt', 'w+', errors="ignore", encoding="ISO-8859-1")
    with open(tweetsPath, "rt", encoding="ISO-8859-1") as f:
        for row in f:
            output.write(row)

def applyPreprocessingInTxtFile():

    preprocessor = getPreprocessor()

    return preprocessor.clean_file('../Files/CleanTxtFile.txt')

def createCleanCsvFrom(cleanTxtFile):
    
    readFile = pd.read_csv(cleanTxtFile, sep=",")
    readFile.to_csv(tweetsPath, index=None) 
    os.remove(cleanTxtFile)

def removePunctuations(tweets): # daha sonra hashtagler için # sembolü geri getirilmeli
    
    tweets.text = tweets.text.str.replace('[^\w\s]', '', regex = True)
    
def makeLowercaseTo(tweets):

    tweets['text'] = tweets['text'].apply(lambda text: " ".join(text.lower() for text in text.split()))

def textLemmatization(text):

    lemmatizer = WordNetLemmatizer()
    text = " ".join([lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in nltk.word_tokenize(text)])

    return text

def removeStopwords(text):
    # set stopwords to english

    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]

    stopWordsList = createNewStopWordList()
    filtered_tokens = [token for token in tokens if token not in stopWordsList]
    filtered_text = ' '.join(filtered_tokens)

    return filtered_text

def removeSpecialNames(text):
    # set stopwords to english

    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]

    specialNameList = createNewSpecialNameList()
    filtered_tokens = [token for token in tokens if token not in specialNameList]
    filtered_text = ' '.join(filtered_tokens)

    return filtered_text

def removeNonEnglishWordsFrom(tweets): #kullanmadık

    tweets['text'] = tweets['text'].apply(lambda x:  " ".join(w for w in nltk.wordpunct_tokenize(x) \
         if w.lower() in words or not w.isalpha()))

def textStemming(text): #kullanmadık

    ps = nltk.porter.PorterStemmer()
    text = ' '.join([ps.stem(word) for word in text.split()])

    return text

def dropDuplicate(tweets):

    tweets.drop_duplicates(subset = "text", keep = 'first' , inplace = True)

def saveCsv(tweets,path):

    dropNaFrom(tweets)
    tweets.to_csv(path, index=None)

def addCleanTextToOriginalFile():

    tweets = pd.read_csv(tweetsPath, sep=",", skipinitialspace=True)  # data frame oldu

    filesize = os.path.getsize(cleanTweetsPath)

    if filesize == 0:
        saveCsv(tweets, cleanTweetsPath)
    else:
        original_tweets = pd.read_csv(cleanTweetsPath, sep=",", skipinitialspace=True)  # data frame oldu

        frames = [tweets, original_tweets]
        total_tweets = pd.concat(frames)
        dropDuplicate(total_tweets)
        saveCsv(total_tweets, cleanTweetsPath)
        print(total_tweets.info())

# Main commands

tweets = pd.read_csv(tweetsPath, sep=",", skipinitialspace=True)

dropNaFrom(tweets)
removeNumbersFrom(tweets)
removeRtFrom(tweets)
removeSpacesFrom(tweets)
addQuotesToEndOfTweetText(tweets)
createTxtForPreprocessing()
cleanTxtFile = applyPreprocessingInTxtFile()
createCleanCsvFrom(cleanTxtFile)

tweets = pd.read_csv(tweetsPath, sep=",", skipinitialspace=True)

removePunctuations(tweets)
makeLowercaseTo(tweets)
tweets['text'] = tweets['text'].apply(textLemmatization)
tweets['text'] = tweets['text'].apply(removeStopwords)
tweets['text'] = tweets['text'].apply(removeSpecialNames)
dropDuplicate(tweets)
saveCsv(tweets, tweetsPath)

addCleanTextToOriginalFile()


# removeNonEnglishWordsFrom(tweets) #kullanmadık
# tweets['text'] = tweets['text'].apply(textStemming) #kullanmadık

