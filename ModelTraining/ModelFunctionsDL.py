import pickle

from tensorflow.python.keras.models import model_from_yaml


def saveModel(model, modelName):

    model.save_weights("ModelsDL/" + modelName + "_weights.h5")
    model_json = model.to_json()
    with open("ModelsDL/" + modelName + ".json" , "w") as json_file:
        json_file.write(model_json)


def loadModel(path, modelName):

    json_file = open(path + modelName + ".json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    loaded_model = model_from_yaml(loaded_model_json)
    loaded_model.load_weights(path + modelName + "_weights.h5")

    return loaded_model


def saveTokenizerOfModel(tokenizer,modelName):

    with open("ModelsDL/" + modelName + 'tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)


def loadTokenizerOfModel(path, modelName):

    with open(path + modelName + 'tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    return tokenizer
