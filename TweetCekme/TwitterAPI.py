import tweepy
from TwitterConfig import api

def getTweetsOfUser(username):
    tweet_count = 20
    userTimeline = api.user_timeline(id = username, count = tweet_count)
    for i in userTimeline:
        print(i.text)

def allCountryCodes():
    places = api.trends_available()
    allCountries = {place['name'].lower() : place['woeid'] for place in places }
    return allCountries

def getWoeIdOfCountry(country):
    country = country.lower()
    trends = api.trends_available()
    allWoeIds = allCountryCodes()
    return allWoeIds[country]

def getTweetsOfHashtag(hashtag):
    tweets = api.search(q = hashtag,
                        lang = "tr",
                        result_type = "recent",
                        count = 2)
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
    # fetching the status with extended tweet_mode
    status = api.get_status(tweetId, tweet_mode="extended")

    user = api.get_user(status.user.screen_name)  # espinozac_ @ile başlayan user' in ismini çeker.
    print(user)
