import pandas as pd
import numpy as np
# import matplotlib as plt

tweets = pd.read_csv('ContentOfTweets.csv', sep = ",")

def convertDataTypes():

    tweets['id'] = tweets['id'].astype('string')
    tweets['tweet'] = tweets['tweet'].astype('string')
    tweets['label'] = tweets['label'].astype('category')

    return tweets

convertDataTypes()
print(tweets.info())
