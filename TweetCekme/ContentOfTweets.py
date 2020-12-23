import csv
from TwitterAPI import getTweetById

class ContentOfTweets:

  def __init__(self, id, label, text):
    self.id = id
    self.label = label
    self.text = text

def writeTweetToCSV(tweet):
    with open('ContentOfTweets.csv',
              'a',
              newline='',
              encoding='utf-8') as file:
        fieldnames = ['id', 'text', 'label']
        writer = csv.DictWriter(file, fieldnames = fieldnames)
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
        print(tweet.id, "Index is: ", index)
        writeTweetToCSV(tweet)
        index = index + 1

tweets = []
tweets = readDataSet("clean_tweets_ml.csv")
writeTweetsToCSV(tweets)







