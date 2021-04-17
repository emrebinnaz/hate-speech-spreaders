import pickle

def saveModel(model,modelName):

    with open('ModelsML/' + modelName + '.pkl' , 'wb') as f:
        pickle.dump(model, f)

def loadModel(path, modelName):

    with open(path + modelName + '.pkl', 'rb') as f:
        model = pickle.load(f)

    return model