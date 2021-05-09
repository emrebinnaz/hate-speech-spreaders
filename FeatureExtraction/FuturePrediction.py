import pandas as pd
import joblib

from nltk.stem import WordNetLemmatizer
from nltk.tokenize.toktok import ToktokTokenizer
import ModelTraining.ModelFunctionsML as ModelFunctionsML
import ModelTraining.ModelFunctionsDL as ModelFunctionsDL
from FeatureExtraction.PredictedTweet import PredictedTweet
from ModelTraining.ModelTrainingDL import tokenizer, max_len
from Preprocessing.Stopwords import createNewStopWordList
from Preprocessing.Stopwords import createNewSpecialNameList
from Config.Preprocessor import *
from Config.Lemmatization import *
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

import os

words = set(nltk.corpus.words.words())
toktokTokenizer = ToktokTokenizer()

mlModelsPath = '../ModelTraining/ModelsML/'
dlModelsPath = '../ModelTraining/ModelsDL/'

def textLemmatization(text):

    lemmatizer = WordNetLemmatizer()
    text = " ".join([lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in nltk.word_tokenize(text)])

    return text


def removeStopwords(text):

    tokens = toktokTokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]

    stopWordsList = createNewStopWordList()
    filtered_tokens = [token for token in tokens if token not in stopWordsList]
    filtered_text = ' '.join(filtered_tokens)

    return filtered_text

def removeSpecialNames(text):
    # set stopwords to english

    tokens = toktokTokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]

    specialNameList = createNewSpecialNameList()
    filtered_tokens = [token for token in tokens if token not in specialNameList]
    filtered_text = ' '.join(filtered_tokens)

    return filtered_text


def predictWithML(modelName, newTextList):

    model = ModelFunctionsML.loadModel(mlModelsPath, modelName)
    tfidfVector = joblib.load(open("../Files/TfidfVector.pkl", "rb"))

    tweets = pd.DataFrame({'text': newTextList})

    tweets = generalPreprocessingForPrediction(tweets)

    wordList = tfidfVector.get_feature_names() # columns
    newSample = tfidfVector.transform(tweets['text']) # Transform dictionary features into 2D feature matrix.
    newSampleAsDataFrame = pd.DataFrame(data = newSample.toarray(),
                                        columns = wordList)

    predictWithModelEnsemble(newSampleAsDataFrame,newTextList)
    # predictions = model.predict(newSampleAsDataFrame)



def predictWithDL(modelName, newTextList):

    model = ModelFunctionsDL.loadModel(dlModelsPath, modelName)

    tweets = pd.DataFrame({'text': newTextList})
    tweets = generalPreprocessingForPrediction(tweets)

    texts = tweets['text'].values.tolist()

    tokenizer.fit_on_texts(texts)
    test_samples_token = tokenizer.texts_to_sequences(texts)
    test_samples_tokens_pad = pad_sequences(test_samples_token, maxlen=max_len)

    print(model.predict(x=test_samples_tokens_pad))


def generalPreprocessingForPrediction(tweets):

    tweets['text'] = tweets['text'].apply(lambda text: getPreprocessor().clean(text))
    tweets.text = tweets.text.replace('#', '', regex=True)  # remove hashtag mark
    tweets.text = tweets.text.replace('\d+', '', regex=True)  # remove numbers
    tweets.text = tweets.text.replace("RT", '', regex=True)  # remove RT
    tweets.text = tweets.text.replace('\s+', ' ', regex=True)  # remove blanks
    tweets.text = tweets.text.str.replace('[^\w\s]', '')  # remove punctuations
    tweets['text'] = tweets['text'].apply(lambda text: " ".join(text.lower() for text in text.split()))  # lowercase

    tweets['text'] = tweets['text'].apply(textLemmatization)
    tweets['text'] = tweets['text'].apply(removeStopwords)
    tweets['text'] = tweets['text'].apply(removeSpecialNames)

    return tweets


def predictWithModelEnsemble(newSampleAsDataFrame, newTextList): # will be continue

    predictedTweets = []

    for newText in newTextList:

        predictedTweet = PredictedTweet(newText)
        predictedTweets.append(predictedTweet)

    modelNameList = os.listdir(mlModelsPath)
    modelNameList = ["DecisionTree.pkl", "LogisticRegression.pkl"] # burası değişcek



    for modelName in modelNameList:

        modelName = modelName.replace(".pkl","")
        model = ModelFunctionsML.loadModel(mlModelsPath, modelName)

        predictions = model.predict(newSampleAsDataFrame)

        index = 0

        for prediction in predictions:

            text = predictedTweets[index].text

            if prediction == 'normal':
                predictedTweets[index].numberOfNormalPrediction += 1
            else:
                predictedTweets[index].numberOfHatefulPrediction += 1

            index += 1

    for predictedTweet in predictedTweets:

        print(predictedTweet.toString(), "Tweet is :", predictedTweet.isTweetHateful())






# predictWithDL("LSTM",["hope", "love", "feel energy", "jew", "nigga", "hate bitch"])
predictWithML('LogisticRegression',["hope", "love", "feel energy", "jew", "nigga", "hate bitch"])







