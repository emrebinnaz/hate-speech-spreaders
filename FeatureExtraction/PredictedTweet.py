class PredictedTweet:

    def __init__(self, text, models):
        self.text = text
        self.models = models
        self.rateOfNormalPrediction = 0
        self.rateOfHatefulPrediction = 0

    def toString(self):

        return self.text + "--->" \
               " Normal diyen modeller : " + str(self.models) + "\n" \
               "Hateful prediction rate:  " + str(self.rateOfHatefulPrediction) + "\n" \
               "Normal prediction rate :" + str(self.rateOfNormalPrediction)


    def isTweetHateful(self):

        return self.rateOfHatefulPrediction > self.rateOfNormalPrediction