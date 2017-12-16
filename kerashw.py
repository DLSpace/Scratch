"""
test modules
"""
import argparse
import datetime
import math
import keras
# import numpy as np
from keras import regularizers
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Activation, BatchNormalization, GaussianNoise, Dropout, RepeatVector, ActivityRegularization, SimpleRNN, LSTM
from keras.optimizers import SGD, Adam
import matplotlib.pyplot as plt

ARGPARSER = argparse.ArgumentParser()
ARGPARSER.add_argument("-f", "--fileName", help="model file name to load")
ARGPARSER.add_argument("-a", "--activation", help="activation for hidden layers")
ARGPARSER.add_argument("-l", "--lossFunction", help="loss function")
ARGPARSER.add_argument("-e", "--epochs", help="number of epochs to train", type=int)
ARGPARSER.add_argument("-b", "--batchSize", help="batch size to use for SGD", type=int)
ARGPARSER.add_argument("-n", "--neuronCount", help="number of neurons in a layer", type=int)
ARGPARSER.add_argument("-y", "--layerCount", help="number of layers in the net", type=int)

CMD_ARGS = ARGPARSER.parse_args()

# module level state variables
MODEL = Sequential()

ACTIVATION = 'relu'
if CMD_ARGS.activation is not None:
    ACTIVATION = CMD_ARGS.activation

LAYER_COUNT = 2
if CMD_ARGS.layerCount is not None:
    LAYER_COUNT = CMD_ARGS.layerCount

def initialize_net():
    """
        Initialize model and setup the layers
    """
    # input layer
    MODEL.add(Dense(units=1, input_dim=1, bias_initializer='RandomNormal'))
    MODEL.add(Activation('linear'))

    #hidden layers
    for i in range(0, LAYER_COUNT, 1):
        MODEL.add(RepeatVector(2)) # for simple RNN
        # hidden layer 1
        MODEL.add(LSTM(units=10, use_bias=True,
        bias_initializer='RandomNormal',
        kernel_regularizer=regularizers.l2(0.005),
        # activity_regularizer=regularizers.l1(0.01)
        ))
        # MODEL.add(Dropout(0.005))
        # MODEL.add(ActivityRegularization(l1 = 0.0, l2 = 0.01))
        MODEL.add(Activation(ACTIVATION))

    # output layer
    MODEL.add(Dense(units=1, input_dim=1, bias_initializer='RandomNormal'))
    MODEL.add(Activation('linear'))
    # MODEL.compile(loss='mean_squared_error',
                #   optimizer='sgd', metrics=['accuracy'])


DATA = []
LABELS = []


def initialize_training_set():
    """
    intialize training set and labels
    """
    for i in range(0, 180, 1):
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
    print("saving model : " + file_name)


def test_model():
    """
    Test the model and check how accurate predictions are
    """
    test_data = []
    for i in range(0, 450, 1):
        test_data.extend([[math.radians(i)]])
    pred = MODEL.predict(test_data, batch_size=1)
    fig, ax = plt.subplots()
    plt.ylabel('Predict')
    plt.xlabel(datetime.datetime.now().strftime(LOSS_FUNCTION + "-" + ACTIVATION + " - %H:%M:%S"))
    plt.ioff()
    plt.autoscale(enable=True, axis='both')
    actuals=[]
    for i in range(0, len(pred), 1):
        actuals.append(math.sin(test_data[i][0]))
    ax.plot(test_data, pred, 'b,',label='Prediction')
    ax.plot(test_data, actuals, 'r,',label='Actual')
    ax.legend(loc='lower left', shadow=False, fontsize='small')
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
my_sgd = keras.optimizers.SGD(lr=0.05, momentum=0.0, decay=0.0, nesterov=False)
my_adam = Adam(lr=0.005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

LOSS_FUNCTION="mean_squared_error"
if CMD_ARGS.lossFunction is not None:
    LOSS_FUNCTION = CMD_ARGS.lossFunction

MODEL.compile(loss=LOSS_FUNCTION, optimizer='Nadam', metrics=['categorical_accuracy'])
# Train the model, iterating on the data in batches of 32 samples
history = MODEL.fit(DATA, LABELS, epochs=EPOCHS, batch_size=BATCH_SIZE)
model_save_file_name= "./models/kerasSinModel" + str(history.history['loss'][-1])
test_model()
save_model(model_save_file_name)
