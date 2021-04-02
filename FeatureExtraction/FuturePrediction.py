import pandas as pd
import joblib
from ModelTraining.ModelFunctions import loadModel


def prediction(modelName, newTextList):

    model = loadModel("../ModelTraining/Models/", modelName)

    tfidfVector = joblib.load(open("../Files/TfidfVector.pkl", "rb"))
    # tfidfVector.min_df = 0

    wordList = tfidfVector.get_feature_names() # columns

    newSample = tfidfVector.transform(newTextList) # Transform dictionary features into 2D feature matrix.

    newSampleAsDataFrame = pd.DataFrame(data = newSample.toarray(),
                                        columns = wordList)

    prediction = model.predict(newSampleAsDataFrame)

    print(prediction)


prediction('MultinomialNaiveBayes', ["fuck you fucking fuck abi","I dont hate you. I am lovely guy. I love you"])