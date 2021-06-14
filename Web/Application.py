from TweetAccess.TwitterAPI import *
from DatabaseOperations.Operations import *
from FeatureExtraction.FuturePrediction import *
from datetime import date

TWEET_COUNT_OF_HASHTAG = 2
HASHTAG_COUNT = 2

today = date.today()
today = today.strftime("%d-%m-%Y")


connection = connectDatabase()
cursor = connection.cursor()


def insertHashtagTuples():

    hashtags = getHashtagList("los angeles", HASHTAG_COUNT)
    hashtagList = []

    for hashtag in hashtags:
        hashtagTuple = (hashtag, today)
        hashtagList.append(hashtagTuple)

    insertHashtag(hashtagList, connection, cursor)


def insertTweetOwnerTuples(tweetOwnerList, connection, cursor):

    ownerList = []

    for owner in tweetOwnerList:

        followers = owner.followers_count
        following = owner.friends_count
        image_url = owner.profile_image_url_https
        name = owner.name
        ownerId = owner.id
        type_of_spreader = "NORMAL"
        username = owner.screen_name
        ownerTuple = (ownerId, today, followers, following, image_url, name, type_of_spreader, username)
        ownerList.append(ownerTuple)

    insertTweetOwner(ownerList, connection, cursor)


def insertTweetsOfHashtagTuples(hashtagsFromDB, tweetIds):

    tweetsOfHashtagList = []

    index = 0
    for hashtagTuple in hashtagsFromDB:

        for tweetId in tweetIds[index: index + TWEET_COUNT_OF_HASHTAG]:

            tweetsOfHashtagTuple = (today, hashtagTuple[0], tweetId)
            tweetsOfHashtagList.append(tweetsOfHashtagTuple)

        index = index + TWEET_COUNT_OF_HASHTAG

    insertTweetsOfHashtag(tweetsOfHashtagList)


def insertTweetTuples(hashtagsFromDB):

    tweetList = []
    tweetIdList = []
    tweetOwnerList = []

    for hashtagTuple in hashtagsFromDB:

        tweets = getTweetsOfHashtag(hashtagTuple[2], TWEET_COUNT_OF_HASHTAG)

        for tweet in tweets:

            tweetId = tweet.id
            favoriteCount = tweet.favorite_count
            retweetCount = tweet.retweet_count
            text = tweet.text
            preprocessedText = " "
            placeOfTweet = "STREAM"
            label = predictWithDL("LSTM.h5", [text])[0]
            tweetOwnerId = tweet.user.id

            tweetTuple = (tweetId, today, favoriteCount, label, placeOfTweet, preprocessedText, retweetCount, text, tweetOwnerId)
            tweetList.append(tweetTuple)
            tweetIdList.append(tweetId)
            tweetOwnerList.append(tweet.user)

    print(tweetIdList)

    insertTweetOwnerTuples(tweetOwnerList, connection, cursor)

    getOwners()

    insertTweet(tweetList)

    insertTweetsOfHashtagTuples(hashtagsFromDB, tweetIdList)


insertHashtagTuples()
hashtagsFromDB = getHashtags(today, connection, cursor)
insertTweetTuples(hashtagsFromDB)

connection.close()