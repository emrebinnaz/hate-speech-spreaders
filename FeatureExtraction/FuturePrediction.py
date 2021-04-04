import nltk
import pandas as pd
import joblib
import preprocessor as preprocessor

from nltk.stem import WordNetLemmatizer
from nltk.tokenize.toktok import ToktokTokenizer
from ModelTraining.ModelFunctions import loadModel
from Preprocessing.Stopwords import createNewStopWordList

words = set(nltk.corpus.words.words())
tokenizer = ToktokTokenizer()

def textLemmatization(text):

    lemmatizer = WordNetLemmatizer()
    text = " ".join([lemmatizer.lemmatize(w) for w in nltk.word_tokenize(text)])

    return text

def removeStopwords(text):

    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]

    stopWordsList = createNewStopWordList()
    filtered_tokens = [token for token in tokens if token not in stopWordsList]
    filtered_text = ' '.join(filtered_tokens)

    return filtered_text


def prediction(modelName, newTextList):

    model = loadModel("../ModelTraining/Models/", modelName)
    tfidfVector = joblib.load(open("../Files/TfidfVector.pkl", "rb"))

    data = {'text': newTextList}
    tweets = pd.DataFrame(data = data)

    preprocessor.set_options(preprocessor.OPT.MENTION,
                             preprocessor.OPT.URL,
                             preprocessor.OPT.RESERVED,
                             preprocessor.OPT.EMOJI,
                             preprocessor.OPT.SMILEY)

    tweets['text'] = tweets['text'].apply(lambda text : preprocessor.clean(text))
    tweets.text = tweets.text.replace('#', '', regex=True)  # remove hashtag mark
    tweets.text = tweets.text.replace('\d+', '', regex=True) # remove numbers
    tweets.text = tweets.text.replace("RT", '', regex=True) # remove RT
    tweets.text = tweets.text.replace('\s+', ' ', regex=True) # remove blanks
    tweets.text = tweets.text.str.replace('[^\w\s]', '') # remove punctuations
    tweets['text'] = tweets['text'].apply(lambda text: " ".join(text.lower() for text in text.split())) # lowercase

    tweets['text'] = tweets['text'].apply(textLemmatization)
    tweets['text'] = tweets['text'].apply(removeStopwords)

    wordList = tfidfVector.get_feature_names() # columns
    newSample = tfidfVector.transform(tweets['text']) # Transform dictionary features into 2D feature matrix.
    newSampleAsDataFrame = pd.DataFrame(data = newSample.toarray(),
                                        columns = wordList)

    prediction = model.predict(newSampleAsDataFrame)
    print(tweets)
    print(prediction)

prediction('MultinomialNaiveBayes', ["farmer ability","A lot of niggas think I’m dumb ima let em think what they want"])