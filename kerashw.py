"""
test modules
"""
import argparse
import datetime
import math
# import numpy as np
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Activation
import matplotlib.pyplot as plt

ARGPARSER = argparse.ArgumentParser()
ARGPARSER.add_argument("-f", "--fileName", help="model file name to load")
ARGPARSER.add_argument("-a", "--activation", help="activation for hidden layers")
ARGPARSER.add_argument("-e", "--epochs", help="number of epochs to train", type=int)
ARGPARSER.add_argument("-b", "--batchSize", help="batch size to use for SGD", type=int)
ARGPARSER.add_argument("-n", "--neuronCount", help="number of neurons in a layer", type=int)
ARGPARSER.add_argument("-y", "--layerCount", help="number of layers in the net", type=int)

CMD_ARGS = ARGPARSER.parse_args()

# module level state variables
MODEL = Sequential()

ACTIVATION='relu'
if CMD_ARGS.activation is not None:
    ACTIVATION = CMD_ARGS.activation

def initialize_net():
    """
        Initialize model and setup the layers
    """
    # input layer
    MODEL.add(Dense(units=1, input_dim=1))
    MODEL.add(Activation('linear'))

    #hidden layers
    for i in range(0, 2, 1):
        # hidden layer 1
        MODEL.add(Dense(units=10, bias_initializer='RandomNormal'))
        MODEL.add(Activation(ACTIVATION))

    # output layer
    MODEL.add(Dense(units=1))
    MODEL.add(Activation('linear'))
    # MODEL.compile(loss='mean_squared_error',
                #   optimizer='sgd', metrics=['accuracy'])


DATA = []
LABELS = []


def initialize_training_set():
    """
    intialize training set and labels
    """
    for i in range(0, 180, 2):
        DATA.extend([[math.radians(i)]])
        LABELS.extend([[math.sin(math.radians(i))]])


def save_model(file_name):
    """
    Saves a trained model definition and weights to file
    """
    model_json = MODEL.to_json()
    with open(file_name + ".json", "w") as json_file:
        json_file.write(model_json)
    MODEL.save_weights(file_name + ".h5")


def test_model():
    """
    Test the model and check how accurate predictions are
    """
    test_data = []
    for i in range(0, 360, 1):
        test_data.extend([[math.radians(i)]])
    pred = MODEL.predict(test_data, batch_size=1)
    fig, ax = plt.subplots()
    plt.ylabel('Predict')
    plt.xlabel(datetime.datetime.now().strftime(ACTIVATION + " - %H:%M:%S"))
    plt.ioff()
    plt.autoscale(enable=True, axis='both')
    for i in range(0, len(pred), 1):
        # ax.plot(epoch,self.avgCost,'.');
        ax.plot(test_data[i][0], pred[i][0], 'b,')
        ax.plot(test_data[i][0], math.sin(test_data[i][0]), 'r,')
        # print('sin(' + str(test_data[i][0]) + ') = ' +
        # str(pred[i][0]) + ' :act: ' + str(math.sin(test_data[i][0])))
    plt.savefig("kerastest.png")
    plt.close()

def loadModel(fileName):
    json_file = open(fileName + ".json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    # print(loaded_model_json)
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(fileName + ".h5")
    return loaded_model
    


if CMD_ARGS.fileName is not None:
    MODEL = loadModel(CMD_ARGS.fileName)
    # print(CMD_ARGS.fileName + " specified")
else:
    initialize_net()

EPOCHS = 2000
if CMD_ARGS.epochs is not None:
    EPOCHS = CMD_ARGS.epochs

BATCH_SIZE=5
if CMD_ARGS.batchSize is not None:
    BATCH_SIZE = CMD_ARGS.batchSize

initialize_training_set()
MODEL.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])
# Train the model, iterating on the data in batches of 32 samples
MODEL.fit(DATA, LABELS, epochs=EPOCHS, batch_size=BATCH_SIZE)
test_model()
save_model("kerasSinModel")
