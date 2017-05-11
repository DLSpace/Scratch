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
saves = [];

fileName = 'model.zip'
if args.fileName is not None:
	fileName = args.fileName;
	saves.extend([fileName]);

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
	global ANN,learningRate, patienceThreshold, fileName;
	ANN = Net.loadModel(fileName)
	ANN.patienceThreshold = patienceThreshold;
	ANN.setLearningRate(learningRate);
	#ANN.computeExpressions();

def combineNets(fileName):
	global ANN,learningRate, patienceThreshold;
	tempANN = Net.loadModel(fileName);
	ol = ANN.net.pop();
	tempANN.net.pop();
	tempANN.net.remove(tempANN.getInputLayer());
	ANN.net.extend(tempANN.net);
	ANN.net.extend([ol]);
	ANN.layerCount = len(ANN.net)-2;
	#adjust layer names
	for i in range(1,len(ANN.net)-1):
		ANN.net[i].name = 'l'+str(i);
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
	global ANN,fileName, saves
	print('')
	print('========================================================')
	print(saves);
	print('\t1.Train')
	print('\t2.Test')
	print('\t3.View')
	print('\t4.Turn Plot', str(not ANN.showPlot));
	print('\t5.Revert')
	print('\t6.Load file')
	print('\t7.Save file')
	print('\t8.Combine nets')
	print('\t9.Exit')
	print('')
	choice = int(input('Choice : '))
	if choice == 1: # Train
		learningRate = float(input('Learning rate : '));
		ANN.setLearningRate(learningRate);
		ANN.patienceThreshold = int(input('Patience : '));
		ANN.train();
		saves.extend([ ANN.saveModel()]);
		args.fileName = saves[-1];
	if choice == 2: #Test
		start = int(input('Start angle : '));
		end = int(input('End angle : '));
		plt.ion()
		fig, ax = plt.subplots()
		plt.ylabel('Predict');
		plt.show();
		plt.autoscale(enable=True,axis='both');
		for ang in range(start,end,1):
			prediction = ANN.predict(ang);
			actualValue = math.sin(math.radians(ang));
			print(prediction, ',', actualValue);
			ax.plot(ang,prediction,'b,');
			ax.plot(ang,actualValue,'r,');
			plt.pause(0.0001);
			plt.draw();
	if choice == 3: #view model
		ANN.printNet();
	if choice ==4:
		ANN.showPlot = not ANN.showPlot;
	if choice ==5:#revert
		if len(saves)>0:
			fileName = saves.pop();
		initialize();
	if choice ==6:
		fileName = input('Enter file name : ')
		args.fileName = fileName;
		initialize();
	if choice ==7:
		ANN.saveModel();
	if choice == 8 :
		fileName = input('Enter file name : ')
		combineNets(fileName);
	if choice == 9: #exit
		exit(0);


initialize();
while True:
	try:
		displayMenu()
	except KeyboardInterrupt:
		print('.............user interrupt');


