import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, cross_val_score
import time
import umap
import matplotlib as plt
from sklearn.tree import DecisionTreeClassifier

originalTweetsPath = '../Files/ContentOfTweets.csv'
trainingSetPath = '../Files/Word2VecTrainingSet.csv'
word2vec_filename = '../Files/train_word2vec.csv'
original_tweets = pd.read_csv(originalTweetsPath, sep=",", skipinitialspace=True)


def tokenizeTweets(original_tweets):

    original_tweets['tokenized_text'] = [simple_preprocess(line, deacc=True) for line in
                                         original_tweets['text']]  # for word2vec


def prepareDataSet(original_tweets):

    numberOfHateful = len(original_tweets[original_tweets['label'] == 'hateful'])
    numberOfNormal = len(original_tweets[original_tweets['label'] == 'normal'])

    minimum = min(numberOfHateful, numberOfNormal)
    print("minimum olan labelın değeri = ", minimum)

    hateful_tweets = getFirstXTweetsOfTargetValue(minimum, 'hateful')
    normal_tweets = getFirstXTweetsOfTargetValue(minimum, 'normal')

    frames = [hateful_tweets, normal_tweets]

    tweets = pd.concat(frames)
    tweets.to_csv(trainingSetPath, index=None)

    return tweets


def getFirstXTweetsOfTargetValue(x, target):

    firstNTweets = original_tweets[original_tweets['label'] == target]
    return firstNTweets.head(n = x)


def split_train_test(test_size=0.25, shuffle_state=True):

    dataSet = pd.read_csv(trainingSetPath,
                          sep=",",
                          skipinitialspace=True)

    X_train, X_test, Y_train, Y_test = train_test_split(dataSet['tokenized_text'],
                                                        dataSet['label'],
                                                        shuffle=shuffle_state,
                                                        test_size=test_size,
                                                        random_state=15)
    X_train = X_train.reset_index()
    X_test = X_test.reset_index()
    Y_train = Y_train.to_frame()
    Y_train = Y_train.reset_index()
    Y_test = Y_test.to_frame()
    Y_test = Y_test.reset_index()

    return X_train, X_test, Y_train, Y_test


def createWord2VecModelFile():

    size = 1000
    window = 5
    min_count = 1
    workers = 3
    sg = 1  # if 1 -> skip-gram if 0 -> cbow

    word2vec_model_file = '../Files/word2vec_' + str(size) + '.model'
    tokenized_text = pd.Series(original_tweets['tokenized_text']).values
    # Train the Word2Vec Model
    w2v_model = Word2Vec(tokenized_text,
                         min_count=min_count,
                         workers=workers,
                         window=window,
                         sg=sg)

    w2v_model.save(word2vec_model_file)


def generateWord2VecVectors(X_train):

    # Store the vectors for train data in following file

    word2vec_model_file = '../Files/word2vec_1000.model'
    sg_w2v_model = Word2Vec.load(word2vec_model_file)

    with open(word2vec_filename, 'w+') as word2vec_file:
        for index, row in X_train.iterrows():
            print("ROWW")
            print(row)
            model_vector = (np.mean([sg_w2v_model[token] for token in row['tokenized_text']], axis=0)).tolist()
            if index == 0:
                header = ",".join(str(ele) for ele in range(1000))
                word2vec_file.write(header)
                word2vec_file.write("\n")
            # Check if the line exists else it is vector of zeros
            if type(model_vector) is list:
                line1 = ",".join([str(vector_element) for vector_element in model_vector])
            else:
                line1 = ",".join([str(0) for i in range(1000)])
            word2vec_file.write(line1)
            word2vec_file.write('\n')

# # Call the train_test_split
# tokenizeTweets(original_tweets)
# dataSet = prepareDataSet(original_tweets)
X_train, X_test, Y_train, Y_test = split_train_test()
# createWord2VecModelFile()
# generateWord2VecVectors(X_train)

word2vec_model_file = '../Files/word2vec_1000.model'
model = Word2Vec.load(word2vec_model_file)


# word2vec_df = pd.read_csv(word2vec_filename)

# clf_decision_word2vec = DecisionTreeClassifier()
# start_time = time.time()
# # Fit the model
# clf_decision_word2vec.fit(df, Y_train['label'])
# print("Time taken to fit the model with word2vec vectors: " + str(time.time() - start_time))
# cross_val_score(df, X_train, Y_train, cv=10)
# predicted = df.predict(X_test)
# print(predicted)
# print(confusion_matrix(Y_test, predicted))
# print(classification_report(Y_test, predicted))
