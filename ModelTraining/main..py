import pandas as pd
from ModelFunctions import *
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import numpy as np

tfidfPath = 'Files/tfidf.csv'
trainingSetPath = 'Files/TrainingSet.csv'

tfidf = pd.read_csv(tfidfPath, sep=",", skipinitialspace=True)

def convertDataTypeToCategoric(df):

    df['label'] = df['label'].astype('category')
    return df

def getFirstXTweetsOfTargetValue(x, target):

    firstNTweets = tfidf[tfidf['label'] == target]
    return firstNTweets.head(n = x)

def prepareTrainingSet():

    hateful_tweets = getFirstXTweetsOfTargetValue(2275,'hateful')
    normal_tweets = getFirstXTweetsOfTargetValue(2275, 'normal')
    frames = [hateful_tweets, normal_tweets]
    tweets = pd.concat(frames)
    tweets.to_csv(trainingSetPath, index=None)

    return tweets

def applyNaiveBayesTo(training_set):

    X_train, X_test, y_train, y_test = train_test_split(training_set.drop(['label'],axis = 1),
                                                        training_set['label'],
                                                        test_size=0.25,
                                                        random_state=42)

    model = MultinomialNB().fit(X_train, y_train)
    saveModel(model, 'naivebayes')

    # K-FOLD INCELENECEK
    # cross_val_score(model, X_train, y_train, cv = 10)

    predicted = model.predict(X_test)
    print(confusion_matrix(y_test, predicted))
    print(classification_report(y_test, predicted))

    # VALIDATION INCELENECEK
    # X_train, X_val, y_train, y_val = train_test_split(X_train,
    #                                                   y_train,
    #                                                   test_size=0.25,
    #                                                   random_state=42)  # 0.25 x 0.8 = 0.2
    #
    # model = MultinomialNB().fit(X_train, y_train)
    # predicted = model.predict(X_val) # validasyon setiyle birlikte?
    #
    # print(confusion_matrix(y_val, predicted))
    # print(accuracy_score(y_val, predicted))

def applyKnnTo(training_set):

    X_train, X_test, y_train, y_test = train_test_split(training_set.drop(['label'], axis=1),
                                                        training_set['label'],
                                                        test_size=0.25,
                                                        random_state=42)

    classifier = KNeighborsClassifier(n_neighbors=2)
    model = classifier.fit(X_train, y_train)
    saveModel(model, 'knn')

    # K-FOLD INCELENECEK
    # cross_val_score(model, X_train, y_train, cv = 5)

    predicted = classifier.predict(X_test)
    print(confusion_matrix(y_test, predicted))
    print(classification_report(y_test, predicted))

    # error = []
    # # Calculating error for K values between 1 and 40
    # for i in range(1, 60):
    #     knn = KNeighborsClassifier(n_neighbors=i)
    #     knn.fit(X_train, y_train)
    #     pred_i = knn.predict(X_test)
    #     error.append(np.mean(pred_i != y_test))
    # a = 0
    # a = error.index(min(error))
    # print(a)

def applyDesicionTreeTo(training_set):

    X_train, X_test, y_train, y_test = train_test_split(training_set.drop(['label'], axis=1),
                                                        training_set['label'],
                                                        test_size=0.25,
                                                        random_state=42)

    classifier = DecisionTreeClassifier()
    model = classifier.fit(X_train, y_train)
    saveModel(model, 'decisiontree')

    # K-FOLD INCELENECEK
    # cross_val_score(model, X_train, y_train, cv = 5)

    predicted = model.predict(X_test)
    print(confusion_matrix(y_test, predicted))
    print(classification_report(y_test, predicted))

training_set = prepareTrainingSet()
training_set = convertDataTypeToCategoric(training_set)
# applyNaiveBayesTo(training_set)
# applyKnnTo(training_set)
applyDesicionTreeTo(training_set)



