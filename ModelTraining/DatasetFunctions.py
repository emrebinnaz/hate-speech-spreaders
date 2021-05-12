import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


def convertDataTypeToCategoric(df):

    df['label'] = df['label'].astype('category')
    return df


def convertLabelToFloat(df):

    df['label'] = df['label'].replace(["hateful", "normal"], [float(1), float(0)])
    df['label'] = pd.to_numeric(df['label'], errors='coerce')

    return df


def prepareDataSetForDL(df):

    numberOfHateful = len(df[df['label'] == 1])
    numberOfNormal = len(df[df['label'] == 0])

    minimum = min(numberOfHateful, numberOfNormal)
    print("minimum olan labelın değeri = ", minimum)

    hateful_tweets = getFirstXTweetsOfTargetValue(minimum, 1, df)
    normal_tweets = getFirstXTweetsOfTargetValue(minimum, 0, df)

    frames = [hateful_tweets, normal_tweets]

    tweets = pd.concat(frames)

    return tweets


def getFirstXTweetsOfTargetValue(x, target, df):

    tweets = df[df['label'] == target]

    return tweets.head(n=x)


def split_train_test(dataSet, test_size=0.25, shuffle_state=True):

    X_train, X_test, Y_train, Y_test = train_test_split(dataSet['text'].values,
                                                        dataSet['label'].values,
                                                        shuffle=shuffle_state,
                                                        test_size=test_size,
                                                        random_state=15)

    return X_train, X_test, Y_train, Y_test


def printConfusionMatrix(y_test, predicted):

    print(confusion_matrix(y_test, predicted))
    print(classification_report(y_test, predicted))