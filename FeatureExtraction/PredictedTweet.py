class PredictedTweet:

    def __init__(self, text):
        self.text = text
        self.numberOfNormalPrediction = 0
        self.numberOfHatefulPrediction = 0

    def toString(self):

        return self.text + \
               " Hateful prediction count:  " + str(self.numberOfHatefulPrediction) + \
               " Normal prediction count :" + str(self.numberOfNormalPrediction)

    def isTweetHateful(self):

        return self.numberOfHatefulPrediction > self.numberOfNormalPrediction