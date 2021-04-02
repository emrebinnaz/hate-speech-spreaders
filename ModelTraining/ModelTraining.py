import pandas as pd
from ModelFunctions import *
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


tfidfPath = '../Files/tfidf.csv'
trainingSetPath = '../Files/TrainingSet.csv'

tfidf = pd.read_csv(tfidfPath, sep=",", skipinitialspace=True)

def convertDataTypeToCategoric(df):

    df['label'] = df['label'].astype('category')
    return df

def getFirstXTweetsOfTargetValue(x, target):

    firstNTweets = tfidf[tfidf['label'] == target]
    return firstNTweets.head(n = x)

def prepareDataSet():

    numberOfHateful = len(tfidf[tfidf['label'] == 'hateful'])
    numberOfNormal = len(tfidf[tfidf['label'] == 'normal'])

    minimum = min(numberOfHateful,numberOfNormal)
    print("minimum olan labelın değeri = ", minimum)

    hateful_tweets = getFirstXTweetsOfTargetValue(minimum,'hateful')
    normal_tweets = getFirstXTweetsOfTargetValue(minimum, 'normal')

    frames = [hateful_tweets, normal_tweets]

    tweets = pd.concat(frames)
    tweets.to_csv(trainingSetPath, index=None)

    return tweets

def applyNaiveBayes():

    model = MultinomialNB().fit(X_train, y_train)
    saveModel(model, 'MultinomialNaiveBayes')

    cross_val_score(model, X_train, y_train, cv = 10)

    predicted = model.predict(X_test)
    print(confusion_matrix(y_test, predicted))
    print(classification_report(y_test, predicted))

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

    classifier = KNeighborsClassifier(n_neighbors=2)
    model = classifier.fit(X_train, y_train)
    saveModel(model, 'Knn')

    cross_val_score(model, X_train, y_train, cv=10)

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

def applyDecisionTree():

    classifier = DecisionTreeClassifier()
    model = classifier.fit(X_train, y_train)
    saveModel(model, 'DecisionTree')

    cross_val_score(model, X_train, y_train, cv=10)

    predicted = model.predict(X_test)
    print(predicted)
    print(confusion_matrix(y_test, predicted))
    print(classification_report(y_test, predicted))


dataSet = prepareDataSet()
dataSet = convertDataTypeToCategoric(dataSet)

X_train, X_test, y_train, y_test = train_test_split(dataSet.drop(['label'],axis = 1),
                                                    dataSet['label'],
                                                    test_size=0.25,
                                                    random_state=42)

# applyNaiveBayes()
# applyKnn()
# applyDecisionTree()
