import pickle


def saveTokenizerOfModel(tokenizer,modelName):

    with open("ModelsDL/" + modelName + 'tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)


def loadTokenizerOfModel(path, modelName):

    with open(path + modelName + 'tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    return tokenizer
