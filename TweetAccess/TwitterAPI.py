import tweepy
from TweetAccess.TwitterConfig import api


def allCountryCodes():

    places = api.trends_available()
    allCountries = {place['name'].lower() : place['woeid'] for place in places }
    return allCountries


def getWoeIdOfCountry(country):

    country = country.lower()
    trends = api.trends_available()
    allWoeIds = allCountryCodes()

    return allWoeIds[country]


def getHashtagList(place, hashtagCount):

    woeid = getWoeIdOfCountry(place)

    # fetching the trends
    trends = api.trends_place(id=woeid)

    hashtagNameList = []
    for value in trends:
        for trend in value['trends']:
            hashtagNameList.append(trend['name'])

    return hashtagNameList[0:hashtagCount]


def getTweetsOfHashtag(hashtag, tweetCount):

    tweets = api.search(q = hashtag,
                        lang = "en",
                        result_type = "recent",
                        tweet_mode='extended',
                        count = tweetCount)
    return tweets


def getTweetsOfUser(username, tweetCount):

    return api.user_timeline(id = username, count = tweetCount, tweet_mode='extended')


def getTweetById(id):

    tweetText = ""
    try:
        status = api.get_status(id, tweet_mode="extended")
        tweetText = status.full_text
        print("The text of the status is : " + tweetText)
    except tweepy.TweepError as error:
        print(error)

    return tweetText


def getUserInformationsOfTweet(tweetId):
    status = api.get_status(tweetId, tweet_mode="extended")

    user = api.get_user(status.user.screen_name)  # espinozac_ @ile başlayan user' in ismini çeker.

    return user


tweets = getTweetsOfHashtag("Hungary", 10)


def getTextOfTweet(tweet):

    if tweet.full_text.startswith("RT @"):

        rtAndUsernamePartOfTweet = tweet.full_text.split(':')[0]

        return rtAndUsernamePartOfTweet + ': ' + tweet.retweeted_status.full_text

    else:
        return tweet.full_text


