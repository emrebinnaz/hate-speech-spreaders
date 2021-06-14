import tweepy
from TweetAccess.TwitterConfig import api


def getTweetsOfUser(username, tweetCount):

    return api.user_timeline(id = username, count = tweetCount)


def allCountryCodes():

    places = api.trends_available()
    allCountries = {place['name'].lower() : place['woeid'] for place in places }
    return allCountries


def getWoeIdOfCountry(country):

    country = country.lower()
    trends = api.trends_available()
    allWoeIds = allCountryCodes()

    return allWoeIds[country]


def getTweetsOfHashtag(hashtag, tweetCount):

    tweets = api.search(q = hashtag,
                        lang = "tr",
                        result_type = "recent",
                        count = tweetCount)
    return tweets


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


def getHashtagList(place):

    woeid = getWoeIdOfCountry(place)

    # fetching the trends
    trends = api.trends_place(id=woeid)

    hashtagNameList = []
    for value in trends:
        for trend in value['trends']:
            hashtagNameList.append(trend['name'])

    return hashtagNameList[0:10]


# tweets = getTweetsOfUser("emrebinnaz",20)
#
# for tweet in tweets:
#
#     username = tweet.user.screen_name
#     tweetOwnerId = tweet.user.id
#     following = tweet.user.friends_count
#     followers = tweet.user.followers_count
#     retweetCount = tweet.retweet_count
#     favoriteCount = tweet.favorite_count
#     text = tweet.text
#     tweetId = tweet.id
#
#     profileImageUrl = tweet.user.profile_image_url_https
#     hashtags = tweet.entities['hashtags']



