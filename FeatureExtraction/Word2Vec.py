
import pandas as pd
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from matplotlib import pyplot, pyplot as plt
from numpy import asarray, zeros
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tensorflow.keras.utils import get_file
from gensim.models.keyedvectors import KeyedVectors


import sys
sys.path.append('/home/emre/Desktop/Belgelerim/HateSpeechSpreaders/hate-speech-spreaders/FeatureExtraction')

originalTweetsPath = '../Files/ContentOfTweets.csv'
trainingSetPath = '../Files/Word2VecTrainingSet.csv'
allVectorValuesPath = '../Files/allWord2VecVectorValues.csv'
original_tweets = pd.read_csv(originalTweetsPath, sep=",", skipinitialspace=True, encoding='utf8')
word2vec_model_file_path = '../Files/word2vec_1000.model'

def tokenizeTweets(original_tweets):

    original_tweets['tokenized_text'] = [simple_preprocess(line, deacc=True) for line in
                                         original_tweets['text']]  # for word2vec


def createWord2VecModelFile():

    size = 1000
    window = 5
    min_count = 1
    workers = 5
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
    word2VecModel = Word2Vec.load(word2vec_model_file_path)

    for tokenizedTextList in originalTweets['tokenized_text']:

        temp = pd.DataFrame()  ## initially empty, and empty on every iteration

        for word in tokenizedTextList:

            try:
                word_vec = word2VecModel.wv[word]  ## if present, the following code applies

                temp = temp.append(pd.Series(word_vec),ignore_index=True)

            except:
                pass

        vector = temp.mean()
        vectors = vectors.append(vector, ignore_index=True)  ## added to the empty data frame

    vectors['label'] = original_tweets['label']
    vectors = vectors.fillna(0)
    vectors.to_csv(allVectorValuesPath, index=None)

    return vectors


def createEmbeddingMatrixFromGlove(vocab_size,tokenizer):

    embeddings_index = dict()
    f = open('../FeatureExtraction/glove.6B.100d.txt')

    for line in f:

        values = line.split()
        word = values[0]
        coefs = asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

    f.close()
    print('Loaded %s word vectors.' % len(embeddings_index))

    # create a weight matrix for words in training docs
    embedding_matrix = zeros((vocab_size, 100))
    for word, i in tokenizer.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return embedding_matrix


def visualizeWord2Vec(model):

    labels = []
    tokens = []



    # for word in model.wv.index_to_key:
    #     tokens.append(model.wv[word])
    #     labels.append(word)
    #
    # tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    # new_values = tsne_model.fit_transform(tokens)
    #
    # x = []
    # y = []
    # for value in new_values:
    #     x.append(value[0])
    #     y.append(value[1])
    #
    # plt.figure(figsize=(16, 16))
    # for i in range(len(x)):
    #     plt.scatter(x[i], y[i])
    #     plt.annotate(labels[i],
    #                  xy=(x[i], y[i]),
    #                  xytext=(5, 2),
    #                  textcoords='offset points',
    #                  ha='right',
    #                  va='bottom')
    # plt.show()



if __name__ == '__main__':

    tokenizeTweets(original_tweets)
    createWord2VecModelFile()
    # vectors = generateWord2VecVectors(original_tweets)
