import pickle

def saveModel(model,modelName):

    with open('Models/'+ modelName + '.pkl' , 'wb') as f:
        pickle.dump(model, f)

def loadModel(modelName):

    with open('Models/'+ modelName + '.pkl', 'rb') as f:
        model = pickle.load(f)

    return model