from tensorflow.python.keras.models import model_from_yaml


def saveModel(model, modelName):

    model_yaml = model.to_yaml()
    with open("ModelsDL/" + modelName + ".yaml" , "w") as yaml_file:
        yaml_file.write(model_yaml)


def loadModel(modelName):

    yaml_file = open("ModelsDL/" + modelName + ".yaml" , 'r')
    loaded_model_yaml = yaml_file.read()
    yaml_file.close()
    loaded_model = model_from_yaml(loaded_model_yaml)

    return loaded_model
