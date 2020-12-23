import tweepy
from tweepy import OAuthHandler

apiKey = 'GN4dgBeg8E3iG09vukRoxge0x'
apiSecret = 'e4ATV8qESbZu47sSWKvt3aCYtiCyUpBKJCWdxMDCDj81OYlw83'
accessToken = '2494179549-c0zNqdH4XIGUdQRHNM6sfOHIwQdYjQVHy8bvi9Z'
accessTokenSecret = 'TXMvpgMhenPkVTPuffm259VUBA5l3o54LNP39Jtc4Kn0g'
# Authenticate to Twitter
auth = OAuthHandler(apiKey, apiSecret)
auth.set_access_token(accessToken, accessTokenSecret)

# Create API object
# tweepy kodlari icin config file yaz
api = tweepy.API(auth, wait_on_rate_limit=True)

