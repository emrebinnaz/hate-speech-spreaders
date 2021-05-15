import pandas as pd
import joblib
from nltk.tokenize.toktok import ToktokTokenizer
from tensorflow.python.keras.models import load_model

import ModelTraining.ModelFunctionsML as ModelFunctionsML
import ModelTraining.ModelFunctionsDL as ModelFunctionsDL
from FeatureExtraction.PredictedTweet import PredictedTweet
from ModelTraining.ModelTrainingDL import tokenizer, max_len
from Preprocessing.Preprocessing import generalPreprocessingForPrediction
from Config.Lemmatization import *
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

import os

words = set(nltk.corpus.words.words())
toktokTokenizer = ToktokTokenizer()

mlModelsPath = '../ModelTraining/ModelsML/'
dlModelsPath = '../ModelTraining/ModelsDL/'


def predictWithML(modelName, newTextList):
    print("Predicted by " + modelName + ".....")

    model = ModelFunctionsML.loadModel(mlModelsPath, modelName)
    tfidfVector = joblib.load(open("../Files/TfidfVector.pkl", "rb"))

    tweets = pd.DataFrame({'text': newTextList})

    tweets = generalPreprocessingForPrediction(tweets)

    wordList = tfidfVector.get_feature_names()  # columns

    newSample = tfidfVector.transform(tweets['text'])  # Transform dictionary features into 2D feature matrix.
    newSampleAsDataFrame = pd.DataFrame(data=newSample.toarray(),
                                        columns=wordList)

    predictions = model.predict(newSampleAsDataFrame)

    print(predictions)


def predictWithDL(modelName, newTextList):
    print("Predicted by " + modelName + ".....")

    model = load_model(dlModelsPath + modelName)
    model.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy'])

    tweets = pd.DataFrame({'text': newTextList})
    tweets = generalPreprocessingForPrediction(tweets)

    texts = tweets['text'].values.tolist()

    tokenizer = ModelFunctionsDL.loadTokenizerOfModel(dlModelsPath, "LSTM")
    tokenizer.fit_on_texts(texts)

    test_samples_token = tokenizer.texts_to_sequences(texts)
    test_samples_tokens_pad = pad_sequences(test_samples_token, maxlen=max_len)

    float_values = model.predict(x=test_samples_tokens_pad)
    string_values = []

    for value in float_values:
        if value < 0.6:
            string_values.append("normal")
        else:
            string_values.append("hateful")

    print(string_values)


def predictWithModelEnsemble(newTextList):
    print("Predicted by " + "Model Ensemble Method .....")

    tfidfVector = joblib.load(open("../Files/TfidfVector.pkl", "rb"))

    tweets = pd.DataFrame({'text': newTextList})
    tweets = generalPreprocessingForPrediction(tweets)

    wordList = tfidfVector.get_feature_names()  # columns

    newSample = tfidfVector.transform(tweets['text'])  # Transform dictionary features into 2D feature matrix.
    newSampleAsDataFrame = pd.DataFrame(data=newSample.toarray(),
                                        columns=wordList)

    predictedTweets = []

    for newText in newTextList:
        predictedTweet = PredictedTweet(newText, " ")
        predictedTweets.append(predictedTweet)

    modelNameList = os.listdir(mlModelsPath)
    modelAccList = [91, 93, 75, 93, 92, 86, 90, 92]
    modelInfo = pd.DataFrame({'Name': modelNameList, 'Acc': modelAccList})

    for index, anyModel in modelInfo.iterrows():
        modelName = anyModel['Name'].replace(".pkl", "")
        model = ModelFunctionsML.loadModel(mlModelsPath, modelName)

        predictions = model.predict(newSampleAsDataFrame)

        index = 0
        for prediction in predictions:
            if prediction == 'normal':
                predictedTweets[index].rateOfNormalPrediction += anyModel["Acc"]
                predictedTweets[index].models += modelName + " "
            else:
                predictedTweets[index].rateOfHatefulPrediction += anyModel["Acc"]

            index += 1

    for predictedTweet in predictedTweets:
        value = ""
        if predictedTweet.isTweetHateful():
            value = "Hateful"
        else:
            value = "Normal"

        print(predictedTweet.toString(), "\nTweet is :", value, "\n")


modelName = "GRU.h5"
# modelName = "LogisticRegression"

newTextList = ["hope", "love", "feel energy", "jew", "nigga", "hate bitch", "dont hate", "racism",
               "i dont love to my baby boy"]

# predictWithDL(modelName, newTextList)
# predictWithML(modelName, newTextList)
predictWithModelEnsemble(newTextList)
