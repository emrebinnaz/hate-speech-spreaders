import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from ModelFunctionsML import *
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from DatasetFunctions import getFirstXTweetsOfTargetValue, printConfusionMatrix, convertDataTypeToCategoric
from sklearn.metrics import plot_confusion_matrix

tfidfPath = '../Files/tfidf.csv'
trainingSetPath = '../Files/TrainingSet.csv'

tfidf = pd.read_csv(tfidfPath, sep=",", skipinitialspace=True)


def prepareDataSet():

    numberOfHateful = len(tfidf[tfidf['label'] == 'hateful'])
    numberOfNormal = len(tfidf[tfidf['label'] == 'normal'])

    minimum = min(numberOfHateful,numberOfNormal)
    print("minimum olan labelın değeri = ", minimum)

    hateful_tweets = getFirstXTweetsOfTargetValue(minimum,'hateful', tfidf)
    normal_tweets = getFirstXTweetsOfTargetValue(minimum, 'normal', tfidf)

    frames = [hateful_tweets, normal_tweets]

    tweets = pd.concat(frames)
    tweets.to_csv(trainingSetPath, index=None)

    return tweets


def applyNaiveBayes():

    print("MultinomialNaiveBayes is running")
    model = MultinomialNB().fit(X_train, y_train)
    saveModel(model, 'MultinomialNaiveBayes')

    cross_val_score(model, X_train, y_train, cv = 10)

    predicted = model.predict(X_test)
    printConfusionMatrix(y_test, predicted)


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


def applyKnn():

    print("KNN is running")
    classifier = KNeighborsClassifier(n_neighbors=2)
    model = classifier.fit(X_train, y_train)
    saveModel(model, 'Knn')

    cross_val_score(model, X_train, y_train, cv=10)

    predicted = classifier.predict(X_test)

    printConfusionMatrix(y_test, predicted)

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


def applyDecisionTree():

    print("Decision Tree is running")
    classifier = DecisionTreeClassifier()
    model = classifier.fit(X_train, y_train)
    saveModel(model, 'DecisionTree')

    cross_val_score(model, X_train, y_train, cv=10)

    predicted = model.predict(X_test)

    printConfusionMatrix(y_test, predicted)


def applyLinearSVM():

    print("Linear SVM is running")
    classifier = svm.SVC(kernel='linear')
    model = classifier.fit(X_train, y_train)
    saveModel(model, 'LinearSVM')

    cross_val_score(model, X_train, y_train, cv=10)

    predicted = classifier.predict(X_test)

    printConfusionMatrix(y_test, predicted)


def applyPolynomialSVM():

    print("Polynomial SVM is running")
    classifier = svm.SVC(kernel = 'poly', degree = 8) # degree ayarlanacak.
    model = classifier.fit(X_train, y_train)
    saveModel(model, 'PolynomialSVM')

    cross_val_score(model, X_train, y_train, cv=10)

    predicted = classifier.predict(X_test)

    printConfusionMatrix(y_test, predicted)


def applyGaussianSVM():

    print("Gaussian SVM is running")
    classifier = svm.SVC(kernel = 'rbf') # degree ayarlanacak.
    model = classifier.fit(X_train, y_train)
    saveModel(model, 'GaussianSVM')

    cross_val_score(model, X_train, y_train, cv=10)

    predicted = classifier.predict(X_test)

    printConfusionMatrix(y_test, predicted)


def applySigmoidSVM():

    print("Sigmoid SVM is running")
    classifier = svm.SVC(kernel='sigmoid')
    model = classifier.fit(X_train, y_train)
    saveModel(model, 'SigmoidSVM')

    cross_val_score(model, X_train, y_train, cv=10)

    predicted = classifier.predict(X_test)

    printConfusionMatrix(y_test, predicted)


def applyLogisticRegression():

    print("Logistic Regression is running")
    classifier = LogisticRegression(random_state=0)
    model = classifier.fit(X_train, y_train)
    saveModel(model, 'LogisticRegression')

    cross_val_score(model, X_train, y_train, cv=10)

    predicted = classifier.predict(X_test)

    printConfusionMatrix(y_test, predicted)


def applyRandomForest():

    print("Random Forest is running")
    classifier = RandomForestClassifier(n_estimators=1000, max_leaf_nodes=18, random_state=21) # parametrelere bak
    model = classifier.fit(X_train, y_train)
    saveModel(model, 'RandomForest')

    cross_val_score(model, X_train, y_train, cv=10)

    predicted = classifier.predict(X_test)

    printConfusionMatrix(y_test, predicted)

    plot_confusion_matrix(model, X_test, y_test)
    plt.show()


dataSet = prepareDataSet()
dataSet = convertDataTypeToCategoric(dataSet)

X_train, X_test, y_train, y_test = train_test_split(dataSet.drop(['label'],axis = 1),
                                                    dataSet['label'],
                                                    test_size=0.25,
                                                    random_state=42)

# model = loadModel("ModelsML/", "RandomForest")
# predicted = model.predict(X_test)
# print(confusion_matrix(y_test, predicted))
# print(classification_report(y_test, predicted))
#
# plot_confusion_matrix(model, X_test, y_test)
# plt.show()
#
# applyNaiveBayes()
# applyKnn()
# applyDecisionTree()
# applyLogisticRegression()
# applyRandomForest()
# # applyLinearSVM()
# applySigmoidSVM()
# applyGaussianSVM()
# applyPolynomialSVM()
