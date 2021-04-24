import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score

trainingSetPath = '../Files/Word2VecTrainingSet.csv'
allVectorValuesPath = '../Files/allWord2VecVectorValues.csv'

def convertDataTypeToCategoric(df):

    cols = [i for i in df.columns if i not in ["label"]]

    for col in cols:
        df[col] = df[col].astype('float64')

    df['label'] = df['label'].astype('category')

    return df

def prepareDataSet(vectors):

    numberOfHateful = len(vectors[vectors['label'] == 'hateful'])
    numberOfNormal = len(vectors[vectors['label'] == 'normal'])

    minimum = min(numberOfHateful, numberOfNormal)
    print("minimum olan labelın değeri = ", minimum)

    hateful_tweets = getFirstXTweetsOfTargetValue(minimum, 'hateful',vectors)
    normal_tweets = getFirstXTweetsOfTargetValue(minimum, 'normal',vectors)

    frames = [hateful_tweets, normal_tweets]

    tweets = pd.concat(frames)
    tweets.to_csv(trainingSetPath, index=None)

    return tweets

def getFirstXTweetsOfTargetValue(x, target,docs_vectors):

    tweets = docs_vectors[docs_vectors['label'] == target]

    return tweets.head(n = x)


def split_train_test(dataSet,test_size=0.25, shuffle_state=True):

    X_train, X_test, Y_train, Y_test = train_test_split(dataSet.drop(['label'],axis = 1),
                                                        dataSet['label'],
                                                        shuffle=shuffle_state,
                                                        test_size=test_size,
                                                        random_state=15)

    Y_train = Y_train.to_frame()
    Y_test = Y_test.to_frame()

    print(Y_train.info())
    print(Y_test.info())
    print(len(Y_train[Y_train['label'] == 'hateful']))
    print(len(Y_train[Y_train['label'] == 'normal']))

    return X_train, X_test, Y_train, Y_test

vectors = pd.read_csv(allVectorValuesPath, sep=",", skipinitialspace=True)
vectors = convertDataTypeToCategoric(vectors)

dataSet = prepareDataSet(vectors)
X_train, X_test, Y_train, Y_test = split_train_test(dataSet)
