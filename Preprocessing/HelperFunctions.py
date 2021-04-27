
tweetsPath = '../Files/DirtyDatasets/neutral-positive-negative-tweets.csv' #gelen verisetine göre path değiştir !!!

def change1toHatefuland0toNormal(tweets): #label 0 ve 1 olan veriseti varsa bunu uygulamayı unutma!!!

    tweets['label'].replace({1:"hateful", 0:"normal"},inplace=True)


def preprocessOf27000HatefulTweetFile():

    dirtyFile = open(tweetsPath, "r")
    file = open("../Files/CleanDatasets/DatasetWith27000Hateful.csv", "w")

    dirtyLines = dirtyFile.readlines()

    # Strips the newline character
    count = 0
    commaCount = 0
    flag = True
    for dirtyLine in dirtyLines:
        count += 1
        dirtyLine = dirtyLine.rstrip('\n')
        labelStringForm = ""
        for index,element in enumerate(dirtyLine):
            if element == ',':
                commaCount += 1

            if commaCount == 5 and flag:
                label = dirtyLine[index + 1]
                if label == "2":
                    labelStringForm = "normal"
                else:
                    labelStringForm = "hateful"

                flag = False

            elif commaCount == 6:

                commaCount = 0
                text = dirtyLine[index + 1 : ]
                text = text.replace(",", "")
                file.write(str(count) + "," + text + "," + labelStringForm)
                file.write("\n")
                flag = True
                break


def preprocessNormalTweets():

    dirtyFile = open(tweetsPath, "r")
    file = open("../Files/CleanDatasets/DatasetWith3500Normal.csv", "w+")

    dirtyLines = dirtyFile.readlines()
    positiveCount = 0
    for dirtyLine in dirtyLines:


        dirtyLine = dirtyLine.rstrip('\n')

        for indexOfFirstComma,element in enumerate(dirtyLine):
            if element == ',':

                dirtyTweetText= dirtyLine[indexOfFirstComma + 1:]
                tweetId = dirtyLine[: indexOfFirstComma]

                splittedTweet = dirtyTweetText.split(",")

                splitCount = len(splittedTweet)
                tweetLabel = splittedTweet[splitCount - 1]

                if isTweetPositive(tweetLabel):
                    positiveCount += 1

                    tweetText = extractTweetText(dirtyLine,tweetLabel, indexOfFirstComma)
                    tweetText = tweetText.replace(",","")

                    tweetText = ' '.join(getUniqueWordsFrom(tweetText.split()))
                    print(tweetText)

                    file.write(tweetId + "," + tweetText + ",normal")
                    file.write("\n")

                    break

        if positiveCount == 3500:
            break


def isTweetPositive(label):

    return (label == 'positive')


def extractTweetText(line, tweetLabel, indexOfFirstComma):

    startIndexOfTweetLabel = line.index("," + tweetLabel)
    tweetText = line[indexOfFirstComma + 1: startIndexOfTweetLabel]

    return tweetText


def getUniqueWordsFrom(l):
    ulist = []
    [ulist.append(x) for x in l if x not in ulist]
    return ulist

preprocessNormalTweets()


