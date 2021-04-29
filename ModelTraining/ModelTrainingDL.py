import pandas as pd
from sklearn.model_selection import train_test_split
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Embedding, GRU

from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.preprocessing.text import Tokenizer

originalTweetsPath = '../Files/ContentOfTweets.csv'
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

    return tweets

def getFirstXTweetsOfTargetValue(x, target,docs_vectors):

    tweets = docs_vectors[docs_vectors['label'] == target]

    return tweets.head(n = x)

def split_train_test(dataSet,test_size=0.25, shuffle_state=True):

    dataSet = dataSet.drop(['id'], axis = 1)
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

# vectors = pd.read_csv(allVectorValuesPath, sep=",", skipinitialspace=True)
# vectors = convertDataTypeToCategoric(vectors)
#
# dataSet = prepareDataSet(vectors)
# X_train, X_test, Y_train, Y_test = split_train_test(dataSet)
#
# # print(device_lib.list_local_devices())
#
# model = Sequential()
# model.add(Dense(128, activation='relu', input_dim=100))
# model.add(Dropout(0.7))
# model.add(Dense(1, activation='sigmoid'))
# model.compile(optimizer='adadelta',
#               loss='binary_crossentropy',
#               metrics=['accuracy'])
# model.summary()
#
# Y_train['label'].replace({'hateful': 1, "normal" : 0}, inplace = True)
# Y_test['label'].replace({'hateful': 1, "normal" : 0}, inplace = True)
#
# model.fit(X_train, Y_train, epochs=20, batch_size=50, validation_data=(X_test,Y_test))
#
# loss, accuracy = model.evaluate(X_train, Y_train, verbose=False)
# print("Training Accuracy: {:.4f}".format(accuracy))
# loss, accuracy = model.evaluate(X_test, Y_test, verbose=False)
# print("Testing Accuracy:  {:.4f}".format(accuracy))

original_tweets = pd.read_csv(originalTweetsPath, sep=",", skipinitialspace=True)

original_tweets['label'] = original_tweets['label'].replace(["hateful", "normal"], [float(1), float(0)])
original_tweets['label'] = pd.to_numeric(original_tweets['label'], errors='coerce')
#original_tweets['label'] = original_tweets['label'].astype(float)
print(original_tweets.info())

X_train = original_tweets.loc[:30000, 'text'].values
X_test = original_tweets.loc[30001:, 'text'].values
Y_train = original_tweets.loc[:30000, 'label'].values
Y_test = original_tweets.loc[30001:, 'label'].values

tokenizer_obj = Tokenizer()
total_tweets = original_tweets['text']
tokenizer_obj.fit_on_texts(total_tweets)

max_length = max([len(s.split()) for s in total_tweets])
vocab_size = len(tokenizer_obj.word_index) + 1

X_train_tokens = tokenizer_obj.texts_to_sequences(X_train)
X_test_tokens = tokenizer_obj.texts_to_sequences(X_test)

X_train_pad = pad_sequences(X_train_tokens,maxlen=max_length,padding='post')
X_test_pad = pad_sequences(X_test_tokens,maxlen=max_length,padding='post')

Embedding(vocab_size, 100, input_length=max_length)

model = Sequential()
model.add(Embedding(vocab_size, 100, input_length=max_length))
model.add(GRU(units=32,dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1,activation='sigmoid'))

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(X_train_pad, Y_train, batch_size=128, epochs=25, validation_data=(X_test_pad, Y_test), verbose=2)