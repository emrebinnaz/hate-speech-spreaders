
import pandas as pd
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import svm
from ModelTraining.ModelFunctions import saveModel

originalTweetsPath = '../Files/ContentOfTweets.csv'
trainingSetPath = '../Files/Word2VecTrainingSet.csv'
allVectorValuesPath = '../Files/allWord2VecVectorValues.csv'
original_tweets = pd.read_csv(originalTweetsPath, sep=",", skipinitialspace=True)


def tokenizeTweets(original_tweets):

    original_tweets['tokenized_text'] = [simple_preprocess(line, deacc=True) for line in
                                         original_tweets['text']]  # for word2vec


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
    print(len(Y_train[Y_train['label'] == 'hateful']))
    print(len(Y_train[Y_train['label'] == 'normal']))


    return X_train, X_test, Y_train, Y_test


def convertDataTypeToCategoric(df):

    cols = [i for i in df.columns if i not in ["label"]]

    for col in cols:
        df[col] = df[col].astype('float64')

    df['label'] = df['label'].astype('category')

    return df


def createWord2VecModelFile():

    size = 1000
    window = 5
    min_count = 1
    workers = 3
    sg = 1  # if 1 -> skip-gram if 0 -> cbow

    word2vec_model_file = '../Files/word2vec_' + str(size) + '.model'
    tokenized_text = pd.Series(original_tweets['tokenized_text']).values

    w2v_model = Word2Vec(tokenized_text,
                         min_count=min_count,
                         workers=workers,
                         window=window,
                         sg=sg)

    w2v_model.save(word2vec_model_file)


def generateWord2VecVectors(originalTweets):

    vectors = pd.DataFrame()
    word2vec_model_file = '../Files/word2vec_1000.model'
    word2VecModel = Word2Vec.load(word2vec_model_file)

    for tokenizedTextList in originalTweets['tokenized_text']:

        temp = pd.DataFrame()  ## initially empty, and empty on every iteration

        for word in tokenizedTextList:

            try:
                word_vec = word2VecModel.wv[word]  ## if present, the following code applies
                print(word)
                temp = temp.append(pd.Series(word_vec),ignore_index=True)

            except:
                pass

        vector = temp.mean()
        vectors = vectors.append(vector, ignore_index=True)  ## added to the empty data frame

    vectors['label'] = original_tweets['label']
    vectors = vectors.fillna(0)
    vectors.to_csv(allVectorValuesPath, index=None)

    return vectors

# tokenizeTweets(original_tweets)
# createWord2VecModelFile()
# vectors = generateWord2VecVectors(original_tweets)

vectors = pd.read_csv(allVectorValuesPath, sep=",", skipinitialspace=True)
vectors = convertDataTypeToCategoric(vectors)

dataSet = prepareDataSet(vectors)
X_train, X_test, Y_train, Y_test = split_train_test(dataSet)


classifier = svm.SVC(kernel='linear')
model = classifier.fit(X_train, Y_train)
# saveModel(model, 'SVMLinearWithWord2Vec') ## change path of models

cross_val_score(model, X_train, Y_train, cv=10)
predicted = classifier.predict(X_test)

print(predicted)
print(confusion_matrix(Y_test, predicted))
print(classification_report(Y_test, predicted))