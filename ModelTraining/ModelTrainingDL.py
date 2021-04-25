import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
import pandas as pd
import os
import re
from gensim.models.phrases import Phrases, Phraser
from time import time
import multiprocessing
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
import numpy as np
from sklearn.preprocessing import scale
import keras
import tensorflow as tf
from keras.models import Sequential, Model
from keras import layers
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, Input, Embedding
from keras.layers.merge import Concatenate
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import RegexpTokenizer
from sklearn.metrics import confusion_matrix
from tensorflow.python.client import device_lib
import ctypes

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

    print("HATEFUL SAYISI",len(Y_train[Y_train['label'] == 'hateful']))
    print("NORMAL SAYISI", len(Y_train[Y_train['label'] == 'normal']))

    return X_train, X_test, Y_train, Y_test

vectors = pd.read_csv(allVectorValuesPath, sep=",", skipinitialspace=True)
vectors = convertDataTypeToCategoric(vectors)

dataSet = prepareDataSet(vectors)
X_train, X_test, Y_train, Y_test = split_train_test(dataSet)

# print(device_lib.list_local_devices())

model = Sequential()
model.add(Dense(128, activation='relu', input_dim=100))
model.add(Dropout(0.7))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adadelta',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()

Y_train['label'].replace({'hateful': 1, "normal" : 0}, inplace = True)
Y_test['label'].replace({'hateful': 1, "normal" : 0}, inplace = True)

model.fit(X_train, Y_train, epochs=20, batch_size=50, validation_data=(X_test,Y_test))

loss, accuracy = model.evaluate(X_train, Y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, Y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))