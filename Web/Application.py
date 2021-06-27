from TweetAccess.TwitterAPI import *
from DatabaseOperations.Operations import *
from FeatureExtraction.FuturePrediction import *
from datetime import date

TWEET_COUNT_OF_HASHTAG = 10
TWEET_COUNT_OF_OWNER = 10
HASHTAG_COUNT = 5

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
        is_most_interacted_user = False
        name = owner.name
        ownerId = owner.id
        type_of_spreader = "NORMAL"
        username = owner.screen_name
        ownerTuple = (ownerId, today, followers, following, image_url, is_most_interacted_user, name, type_of_spreader, username)
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

            if tweet.full_text.startswith("RT @"):
                favoriteCount = tweet.retweeted_status.favorite_count
            else:
                favoriteCount = tweet.favorite_count

            retweetCount = tweet.retweet_count
            text = getTextOfTweet(tweet)
            placeOfTweet = "STREAM"
            label = predictWithDL("LSTM.h5", [text])[0]
            tweetOwnerId = tweet.user.id
            created_date = tweet.created_at
            tweetTuple = (tweetId, today, favoriteCount, label, placeOfTweet,
                          retweetCount, text, tweetOwnerId, created_date)
            tweetList.append(tweetTuple)
            tweetIdList.append(tweetId)
            tweetOwnerList.append(tweet.user)

    insertTweetOwnerTuples(tweetOwnerList, connection, cursor)

    insertTweet(tweetList)

    insertTweetsOfHashtagTuples(hashtagsFromDB, tweetIdList)


def insertTweetsOfOwners(mostInteractedTweetOwnerIdList):

    tweetList = []
    for mostInteractedTweetOwnerId in mostInteractedTweetOwnerIdList:

        tweets = getTweetsOfUser(mostInteractedTweetOwnerId, TWEET_COUNT_OF_OWNER)

        for tweet in tweets:

            tweetId = tweet.id

            if tweet.full_text.startswith("RT @"):
                favoriteCount = tweet.retweeted_status.favorite_count
            else:
                favoriteCount = tweet.favorite_count

            retweetCount = tweet.retweet_count
            text = getTextOfTweet(tweet)
            placeOfTweet = "PROFILE"
            label = predictWithDL("LSTM.h5", [text])[0]
            tweetOwnerId = tweet.user.id
            created_date = tweet.created_at

            tweetTuple = (tweetId, today, favoriteCount, label, placeOfTweet,
                          retweetCount, text, tweetOwnerId, created_date)
            tweetList.append(tweetTuple)

    insertTweetForTweetOwners(tweetList)
    updateTweetOwner(mostInteractedTweetOwnerIdList)


def updateTweetOwner(mostInteractedTweetOwnerIdList):

    mostInteractedTweetOwnerIdListTuple = []
    for id in mostInteractedTweetOwnerIdList:
        idTuple = (True, id)
        mostInteractedTweetOwnerIdListTuple.append(idTuple)

    updateTweetOwnerForMostInteractedUser(mostInteractedTweetOwnerIdListTuple)


insertHashtagTuples()
hashtagsFromDB = getHashtags(today, connection, cursor)
insertTweetTuples(hashtagsFromDB)
mostInteractedTweetOwnerIdList = getMostInteractedTweetOwnerIds()
insertTweetsOfOwners(mostInteractedTweetOwnerIdList)
connection.close()