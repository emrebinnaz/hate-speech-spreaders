from tensorflow.python.keras.models import model_from_yaml


def saveModel(model, modelName):

    model.save_weights("ModelsDL/" + modelName + "_weights.h5")
    model_yaml = model.to_yaml()
    with open("ModelsDL/" + modelName + ".yaml" , "w") as yaml_file:
        yaml_file.write(model_yaml)


def loadModel(path, modelName):

    yaml_file = open(path + modelName + ".yaml" , 'r')
    loaded_model_yaml = yaml_file.read()
    yaml_file.close()

    loaded_model = model_from_yaml(loaded_model_yaml)
    loaded_model.load_weights(path + modelName + "_weights.h5")

    return loaded_model

