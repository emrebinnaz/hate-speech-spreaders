import pandas as pd
import nltk
import preprocessor as preprocessor
import os

from Stopwords import createNewStopWordList
from nltk.stem import WordNetLemmatizer
from nltk.tokenize.toktok import ToktokTokenizer

words = set(nltk.corpus.words.words())
tokenizer = ToktokTokenizer()

tweetsPath = '../Files/.csv' #gelen verisetine göre path değiştir !!!
cleanTweetsPath = '../Files/ContentOfTweets.csv'

def change1toHatefuland0toNormal(tweets): #label 0 ve 1 olan veriseti varsa bunu uygulamayı unutma!!!

    tweets['label'].replace({1:"hateful", 0:"normal"},inplace=True)

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

    preprocessor.set_options(preprocessor.OPT.MENTION,
                             preprocessor.OPT.URL,
                             preprocessor.OPT.RESERVED,
                             preprocessor.OPT.EMOJI,
                             preprocessor.OPT.SMILEY)

    return preprocessor.clean_file('../Files/CleanTxtFile.txt')

def createCleanCsvFrom(cleanTxtFile):
    
    readFile = pd.read_csv(cleanTxtFile, sep=",")
    readFile.to_csv(tweetsPath, index=None) 
    os.remove(cleanTxtFile)

def removePunctuations(tweets): # daha sonra hashtagler için # sembolü geri getirilmeli
    
    tweets.text = tweets.text.str.replace('[^\w\s]', '')
    
def makeLowercaseTo(tweets):

    tweets['text'] = tweets['text'].apply(lambda text: " ".join(text.lower() for text in text.split()))

def textLemmatization(text):

    lemmatizer = WordNetLemmatizer()
    text = " ".join([lemmatizer.lemmatize(w) for w in nltk.word_tokenize(text)])
    return text

def removeStopwords(text):
    # set stopwords to english

    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]

    stopWordsList = createNewStopWordList()
    filtered_tokens = [token for token in tokens if token not in stopWordsList]
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
    tweets.drop_duplicates(subset="text", keep=False, inplace=True)

def saveCsv(tweets,path):

    dropNaFrom(tweets)
    tweets.to_csv(path, index=None)

def addCleanTextToOriginalFile():

    tweets = pd.read_csv(tweetsPath, sep=",", skipinitialspace=True)  # data frame oldu

    filesize = os.path.getsize(cleanTweetsPath)
    print(filesize)
    if filesize == 0:
        saveCsv(tweets, cleanTweetsPath)
    else:
        original_tweets = pd.read_csv(cleanTweetsPath, sep=",", skipinitialspace=True)  # data frame oldu

        frames = [tweets, original_tweets]
        total_tweets = pd.concat(frames)

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
dropDuplicate(tweets)
saveCsv(tweets,tweetsPath)

addCleanTextToOriginalFile()

removeNonEnglishWordsFrom(tweets) #kullanmadık
tweets['text'] = tweets['text'].apply(textStemming) #kullanmadık