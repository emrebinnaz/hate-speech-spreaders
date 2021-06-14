from TweetAccess.TwitterAPI import *
from DatabaseOperations.Operations import *
from datetime import date

today = date.today()
today = today.strftime("%d-%m-%Y")

connection = connectDatabase()
cursor = connection.cursor()


def createHashtagTuples(hashtags):

    hashtagList = []

    for hashtag in hashtags:
        hashtagTuple = (hashtag, today)
        hashtagList.append(hashtagTuple)

    return hashtagList


hashtags = getHashtagList("london")
hashtagTuples = createHashtagTuples(hashtags)
insert(hashtagTuples, connection, cursor)
connection.close()