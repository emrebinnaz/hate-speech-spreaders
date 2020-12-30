import pandas as pd
from ModelFunctions import *
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import numpy as np

tfidfPath = 'Files/tfidf.csv'
trainingSetPath = 'Files/TrainingSet.csv'

tfidf = pd.read_csv(tfidfPath, sep=",", skipinitialspace=True)

def convertDataTypeToCategoric(tfidf):

    tfidf['label'] = tfidf['label'].astype('category')
    return tfidf


def getFirstXTweetsOfTargetValue(x, target):

    firstNTweets = tfidf[tfidf['label'] == target]
    return firstNTweets.head(n = x)


def prepareTrainingSet():

    hateful_tweets = getFirstXTweetsOfTargetValue(2275,'hateful')
    normal_tweets = getFirstXTweetsOfTargetValue(2275, 'normal')
    frames = [hateful_tweets,normal_tweets]
    tweets = pd.concat(frames) # normal teweetler ve hatefulları birleiştridim
    tweets.to_csv(trainingSetPath, index=None)

    return tweets

def applyNaiveBayesTo(training_set):

    # X_train, X_test, y_train, y_test = train_test_split(training_set.drop(['label'],axis = 1),
    #                                                     training_set['label'],
    #                                                     test_size=0.30)

    X_train, X_test, y_train, y_test = train_test_split(training_set.drop(['label'],axis = 1),
                                                        training_set['label'],
                                                        test_size=0.2,
                                                        random_state=42)
    model = MultinomialNB().fit(X_train, y_train)
    saveModel(model,'naivebayes')
    predicted = model.predict(X_test)

    print(confusion_matrix(y_test, predicted))
    print(accuracy_score(y_test, predicted)) #validasyon seti olmadan

    X_train, X_val, y_train, y_val = train_test_split(X_train,
                                                      y_train,
                                                      test_size=0.25,
                                                      random_state=42)  # 0.25 x 0.8 = 0.2


    model = MultinomialNB().fit(X_train, y_train)
    predicted = model.predict(X_val) # validasyon setiyle birlikte ?

    print(confusion_matrix(y_val, predicted))
    print(accuracy_score(y_val, predicted))


tfidf = convertDataTypeToCategoric(tfidf)
training_set = prepareTrainingSet()
training_set = convertDataTypeToCategoric(training_set)
applyNaiveBayesTo(training_set)
