import pandas as pd
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
originalTweetsPath = '../Files/ContentOfTweets.csv'
trainingSetPath = '../Files/Word2VecTrainingSet.csv'
allVectorValuesPath = '../Files/allWord2VecVectorValues.csv'
original_tweets = pd.read_csv(originalTweetsPath, sep=",", skipinitialspace=True)


def tokenizeTweets(original_tweets):

    original_tweets['tokenized_text'] = [simple_preprocess(line, deacc=True) for line in
                                         original_tweets['text']]  # for word2vec


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

                temp = temp.append(pd.Series(word_vec),ignore_index=True)

            except:
                pass

        vector = temp.mean()
        vectors = vectors.append(vector, ignore_index=True)  ## added to the empty data frame

    vectors['label'] = original_tweets['label']
    vectors = vectors.fillna(0)
    vectors.to_csv(allVectorValuesPath, index=None)

    return vectors

tokenizeTweets(original_tweets)
createWord2VecModelFile()
vectors = generateWord2VecVectors(original_tweets)
