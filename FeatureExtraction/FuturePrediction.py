import pandas as pd
import joblib

from nltk.stem import WordNetLemmatizer
from nltk.tokenize.toktok import ToktokTokenizer
from ModelTraining.ModelFunctionsML import loadModel
from Preprocessing.Stopwords import createNewStopWordList
from Preprocessing.Stopwords import createNewSpecialNameList
from Config.Preprocessor import *
from Config.Lemmatization import *

import os

words = set(nltk.corpus.words.words())
tokenizer = ToktokTokenizer()

mlModelsPath = '../ModelTraining/ModelsML/'

def textLemmatization(text):

    lemmatizer = WordNetLemmatizer()
    text = " ".join([lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in nltk.word_tokenize(text)])

    return text


def removeStopwords(text):

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


def prediction(modelName, newTextList):

    model = loadModel(mlModelsPath, modelName)
    tfidfVector = joblib.load(open("../Files/TfidfVector.pkl", "rb"))

    data = {'text': newTextList}
    tweets = pd.DataFrame(data = data)

    tweets['text'] = tweets['text'].apply(lambda text : getPreprocessor().clean(text))
    tweets.text = tweets.text.replace('#', '', regex=True)  # remove hashtag mark
    tweets.text = tweets.text.replace('\d+', '', regex=True) # remove numbers
    tweets.text = tweets.text.replace("RT", '', regex=True) # remove RT
    tweets.text = tweets.text.replace('\s+', ' ', regex=True) # remove blanks
    tweets.text = tweets.text.str.replace('[^\w\s]', '') # remove punctuations
    tweets['text'] = tweets['text'].apply(lambda text: " ".join(text.lower() for text in text.split())) # lowercase

    tweets['text'] = tweets['text'].apply(textLemmatization)
    tweets['text'] = tweets['text'].apply(removeStopwords)
    tweets['text'] = tweets['text'].apply(removeSpecialNames)

    wordList = tfidfVector.get_feature_names() # columns
    newSample = tfidfVector.transform(tweets['text']) # Transform dictionary features into 2D feature matrix.
    newSampleAsDataFrame = pd.DataFrame(data = newSample.toarray(),
                                        columns = wordList)

    # predictWithModelEnsemble(newSampleAsDataFrame)
    prediction = model.predict(newSampleAsDataFrame)
    print(tweets)
    print(prediction)


prediction('LogisticRegression', ["this place dirty"])


def predictWithModelEnsemble(newSampleAsDataFrame): # will be continue

    modelNameList = os.listdir(mlModelsPath)
    models = []
    index = 0

    for modelName in modelNameList:
        models[index] = loadModel(mlModelsPath, modelName)
        index += 1
