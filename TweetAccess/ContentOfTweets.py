import csv
from TwitterAPI import *

class ContentOfTweets:

  def __init__(self, id, label, text):
    self.id = id
    self.label = label
    self.text = text

def writeTweetToCSV(tweet,index):
    with open('../Files/CleanDatasets/DatasetWithTweetIdFromTweetAccess.csv',
              'a',
              newline='',
              encoding='utf-8') as file:
        fieldnames = ['id', 'text', 'label']
        writer = csv.DictWriter(file, fieldnames = fieldnames)
        if index == 0 :
            writer.writerow({'id': "id",
                             'text': "text",
                             'label': "label"})
        writer.writerow({'id': tweet.id,
                         'text': tweet.text,
                         'label': tweet.label})

def readDataSet(filename):
    with open(filename, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                line_count += 1

            t = ContentOfTweets(row["tweet_id"], row["label"], "")
            tweets.append(t)
            line_count += 1

    return tweets

def removeCommasFrom(tweetText):
    tweetText = tweetText.replace(',', '')

    return tweetText

def writeTweetsToCSV(tweets):
    index = 0
    for tweet in tweets:
        tweet.text = getTweetById(tweet.id)
        tweet.text = removeCommasFrom(tweet.text)

        writeTweetToCSV(tweet,index)
        index = index + 1

tweets = []
tweets = readDataSet("../Files/DirtyDatasets/DatasetWithTweetId.csv")
writeTweetsToCSV(tweets)










