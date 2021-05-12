import pickle
import pandas as pd
import numpy as np
import collections
import tensorflow as tf
import random as python_random
from keras.models import Sequential
from keras.layers import Dense, Embedding, GRU
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.layers import LSTM
from keras import layers
from keras.models import load_model
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.optimizers import RMSprop
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from ModelTraining.DatasetFunctions import convertLabelToFloat, prepareDataSetForDL, split_train_test
from ModelTraining.ModelFunctionsDL import *
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from numpy.random import seed


originalTweetsPath = '../Files/ContentOfTweets.csv'
allVectorValuesPath = '../Files/allWord2VecVectorValues.csv'


max_words = 10000
max_len = 150

tokenizer = Tokenizer(num_words=max_words)

np.random.seed(42)
python_random.seed(42)
seed(42)  # keras seed fixing
tf.random.set_seed(42)  # tensorflow seed fixing


def deepLearningMethodWithWord2Vec():

    print("DL Method with Word2Vec is running")
    word2VecValues = pd.read_csv(allVectorValuesPath, sep=",", skipinitialspace=True)
    word2VecValues = convertLabelToFloat(word2VecValues)
    word2VecValues = prepareDataSetForDL(word2VecValues)

    X = word2VecValues.iloc[:, :-1]
    Y = word2VecValues.iloc[:, -1]

    model = Sequential()
    model.add(Dense(12, input_dim=100, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X, Y, epochs=25, batch_size=10)

    saveModel(model, "Word2Vec")

    _, accuracy = model.evaluate(X, Y)
    print('Accuracy: %.2f' % (accuracy * 100))


def applyGRU(tweets):

    X_train, X_test, Y_train, Y_test = split_train_test(tweets)

    tokenizer_obj = Tokenizer()
    total_tweets = original_tweets['text'].values
    tokenizer_obj.fit_on_texts(total_tweets)

    max_length = max([len(s.split()) for s in total_tweets])
    vocab_size = len(tokenizer_obj.word_index) + 1

    X_train_tokens = tokenizer_obj.texts_to_sequences(X_train)
    X_test_tokens = tokenizer_obj.texts_to_sequences(X_test)

    X_train_pad = pad_sequences(X_train_tokens, maxlen=max_length, padding='post')
    X_test_pad = pad_sequences(X_test_tokens, maxlen=max_length, padding='post')

    print("GRU deep learning method is running")
    Embedding(vocab_size, 100, input_length=max_length)
    model = Sequential()
    model.add(Embedding(vocab_size, 100, input_length=max_length))
    model.add(GRU(units=32, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.fit(X_train_pad, Y_train, batch_size=128, epochs=1, validation_data=(X_test_pad, Y_test), verbose=2)
    modelName = "GRU"
    saveModel(model, modelName)


def applyLSTM(tweets):

    texts = tweets['text']
    labels = tweets['label']

    labels = LabelEncoder().fit_transform(labels)
    labels = labels.reshape(-1, 1) # galiba lstm için boyutları değiştirdi.

    X_train, X_test, Y_train, Y_test = train_test_split(texts, labels, test_size=0.20,random_state=42)

    # print(np.count_nonzero(Y_train))
    # print(len(Y_train) - np.count_nonzero(Y_train))

    tokenizer.fit_on_texts(X_train)
    sequences = tokenizer.texts_to_sequences(X_train)
    sequences_matrix = sequence.pad_sequences(sequences, maxlen=max_len)

    print("LSTM deep learning method is running")
    inputs = Input(name='inputs', shape=[max_len])
    layer = Embedding(max_words, 50, input_length=max_len)(inputs)
    layer = LSTM(64)(layer)
    layer = Dense(256, name='FC1')(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(1, name='out_layer')(layer)
    layer = Activation('sigmoid')(layer)
    model = Model(inputs=inputs, outputs=layer)

    earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0.0001, restore_best_weights=False)

    model.summary()
    model.compile(loss='binary_crossentropy', optimizer = 'adam', metrics=['accuracy'])
    model.fit(sequences_matrix, Y_train, batch_size=128, shuffle=True, epochs=10,
              validation_split=0.2, callbacks=[earlyStopping])

    # validation set'te accuracy artmıyorsa, EarlyStopping sayesinde eğitim duruyor.

    modelName = "LSTM"
    model.save('ModelsDL/' + modelName + ".h5")

    saveTokenizerOfModel(tokenizer, modelName)

    test_sequences = tokenizer.texts_to_sequences(X_test)
    test_sequences_matrix = sequence.pad_sequences(test_sequences, maxlen=max_len)

    accr = model.evaluate(test_sequences_matrix, Y_test)

    print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0], accr[1]))

    tests = ["hope", "feel relax", "feel energy", "peaceful day"]

    tokenizer.fit_on_texts(tests)
    test_samples_token = tokenizer.texts_to_sequences(tests)
    test_samples_tokens_pad = pad_sequences(test_samples_token, maxlen=max_len)

    print(model.predict(x=test_samples_tokens_pad))

    del model

if __name__ == '__main__':

    original_tweets = pd.read_csv(originalTweetsPath, sep=",", skipinitialspace=True)
    original_tweets = convertLabelToFloat(original_tweets)
    original_tweets = prepareDataSetForDL(original_tweets)

    applyLSTM(original_tweets)
    # applyGRU(original_tweets)

    texts = ["hope", "feel relax", "feel energy", "peaceful day", "feel bad"]

    model = load_model("ModelsDL/LSTM.h5")
    model.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy'])

    tokenizer = loadTokenizerOfModel("ModelsDL/", "LSTM")
    tokenizer.fit_on_texts(texts)

    test_samples_token = tokenizer.texts_to_sequences(texts)
    test_samples_tokens_pad = pad_sequences(test_samples_token, maxlen=max_len)

    print(model.predict(x=test_samples_tokens_pad))