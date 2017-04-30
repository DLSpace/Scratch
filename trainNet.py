import argparse
parser = argparse.ArgumentParser()
#parser.add_argument("-l", "--load", help="load saved model",action="store_true")
parser.add_argument("-s", "--noPlot", help="display cost plot",action="store_true")
parser.add_argument("-p", "--patience", help="patience level",type=int)
parser.add_argument("-n", "--neuronCount", help="number of neurons in a layer",type=int)
parser.add_argument("-y", "--layerCount", help="number of layers in the net",type=int)
parser.add_argument("-r", "--learningRate", help="learning rate for the layers",type=float)
parser.add_argument("-f", "--fileName", help="model file name to load")
args = parser.parse_args()

import numpy as np
import theano
import theano.tensor as T
from theano import shared
from theano import function
import theano.printing as pp
import layer
from layer  import Layer
import math
import sys
import matplotlib.pyplot as plt
import six.moves.cPickle as pickle
import shutil
from shutil import copyfile
import net
from net import Net


ANN = Net();

fileName = 'model.zip'
if args.fileName is not None:
	fileName = args.fileName

ANN.showPlot = True;
if args.noPlot:
	ANN.showPlot = False

layerCount = 1;
if args.layerCount is not None:
	layerCount = args.layerCount;

layerNeuronCount = 4;
if args.neuronCount is not None:
	layerNeuronCount = args.neuronCount;

patienceThreshold = 25
if args.patience is not None:
	patienceThreshold = args.patience

learningRate = 0.123
if args.learningRate is not None:
	learningRate = args.learningRate;


def loadModel():
	global ANN,learningRate,layerCount,layerNeuronCount, patienceThreshold;
	ANN = Net.loadModel(fileName)
	ANN.patienceThreshold = patienceThreshold;
	ANN.setLearningRate(learningRate);

def initialize():
	global ANN,layerCount,layerNeuronCount;
	if args.fileName is not None:
		loadModel()
	else:
		ANN.createNet(layerCount,layerNeuronCount);
	ANN.setLearningRate(learningRate);

#testing
def predict(ang):
	rads = math.radians(ang);
	ANN.getInputLayer().activations.set_value(np.array([np.float32(rads)]));
	print('ANN : ',ANN.getOutputLayer().activations.eval(), ' sin : ', math.sin(rads));


def displayMenu():
	global ANN
	print('')
	print('========================================================')
	print('\t1.Train')
	print('\t2.Test')
	print('\t3.View')
	print('\t4.Turn Plot', str(not ANN.showPlot));
	print('\t5.Exit')
	print('')
	choice = int(input('Choice : '))
	if choice == 1: # Train
		inputVals = np.arange(start=0,stop=270,step=2);
		learningRate = float(input('Learning rate : '));
		ANN.setLearningRate(learningRate);
		ANN.patienceThreshold = int(input('Patience : '));
		ANN.train(inputVals);
		ANN.saveModel();
	if choice == 2: #Test
		start = int(input('Start angle : '));
		end = int(input('End angle : '));
		for ang in range(start,end,1):
			print(ANN.predict(ang), ',', math.sin(math.radians(ang)));
	if choice == 3: #view model
		ANN.printNet();
	if choice ==4:
		ANN.showPlot = not ANN.showPlot;
	if choice == 5: #exit
		exit(0);


initialize();
while True:
	displayMenu()


