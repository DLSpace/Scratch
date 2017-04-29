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

il = None;
ol = None;
avgCost = None;
SGDS = [];

fileName = 'model.zip'
if args.fileName is not None:
	fileName = args.fileName
showPlot = True;
if args.noPlot:
	showPlot = False
net = [];
layerCount = 1;
if args.layerCount is not None:
	layerCount = args.layerCount;
layerNeuronCount=10;
if args.neuronCount is not None:
	layerNeuronCount = args.neuronCount;
patienceThreshold = 25
if args.patience is not None:
	patienceThreshold = args.patience
learningRate = 2.5
if args.learningRate is not None:
	learningRate = args.learningRate

def dumpModel():
	for l in net:
		print('-------------------------------');
		if hasattr(l,'name'):
			print('Layer name ',l.name)
		if hasattr(l,'weights'):
			print(l.weights.eval())
		if hasattr(l,'biases'):
			print(l.biases.eval())
		print('-------------------------------');

def loadModel():
	global net,il,ol;
	print('loading existing model.')	
	model = open(fileName,'rb');
	net = pickle.load(model);
	model.close()
	for l in net:
		l.learningRate = learningRate
	il = net[0];
	ol = net[len(net)-1];
	layerCount = len(net)-2;#substract input and out put layers
	layerNeuronCount = net[1].neuronCount;
	#dumpModel()

def createModel():
	global net,il,ol;
	print('intializing model.')
	il = Layer(neurons=1,inputNeurons=0,name='il');
	net.extend([il]);
	for i in range(1,layerCount+1,1):
		pl = net[i-1]; # previous layer
		l1 = Layer(neurons=layerNeuronCount,inputNeurons=pl.neuronCount,inputLayer = pl,name='l'+str(i));
		net.extend([l1])
	ol = Layer(neurons=1,inputNeurons=net[-1].neuronCount,inputLayer = net[-1],name='ol');
	net.extend([ol]);

def initialize():
	global SGDS;
	if args.fileName is not None:
		loadModel()
	else:
		createModel()
	#compute expressions
	il.activations = theano.shared(value=np.array([np.float32(0)]),name='a',allow_downcast=True);
	for l in net:
		l.calculateActivations()
	for l in net:
		if hasattr(l,'weights'):
			SGDS.extend([l.sgd()])

#training
def trainModel():
	global avgCost, showPlot;
	inputVals = np.arange(start=0,stop=270,step=2);
	counter=0
	curcost=[]
	print('Plot flag ',showPlot);
	if showPlot:
		plt.ion()
		fig, ax = plt.subplots()
		plt.ylabel('cost');
		plt.show();
		ax.set_ylim([0,1]);
		ax.set_xlim([0,25]);
		plt.autoscale(enable=True,axis='both');
	prvCost = 0.9;
	patience = 0;
	while(patience <= patienceThreshold):
		for ang in inputVals:
			rads = np.float32(math.radians(ang));
			expected = np.float32(math.sin(rads));
			net[0].activations.set_value(np.array([rads]));
			curcost.extend([SGDS[-1](expected).tolist()])
			for i in range(-1, -len(net),-1):#skips output layer at zero index
				SGDS[i](np.float32(net[i].activations.eval().mean()))
			#l2SGD(np.float32(ol.activations.eval().mean()));
			#l1SGD(np.float32(l2.activations.eval().mean()));
		avgCost = (sum(curcost)/len(curcost));
		prctCost = (((prvCost - avgCost)/avgCost)*100);
		print(avgCost, '  change% :' ,prctCost, 'patience : ' , patience);
		prvCost = avgCost
		if showPlot:
			ax.plot(counter,avgCost,'.');
			plt.pause(0.0001);
			plt.draw();
		if prctCost<=0.001 :
			patience = patience + 1
		else:
			patience = 0;	
		curcost.clear();
		counter = counter+1

def saveModel():
	model = open('model-L'+str(layerCount)+'-N'+str(layerNeuronCount)+'-R'+str(learningRate)+'-P'+str(patienceThreshold)+'-'+str(avgCost)+'.zip','wb');
	pickle.dump(net,model,protocol=pickle.HIGHEST_PROTOCOL);
	model.close()

#testing
def predict(ang):
	rads = math.radians(ang);
	il.activations.set_value(np.array([np.float32(rads)]));
	print('ANN : ',ol.activations.eval(), ' sin : ', math.sin(rads));

initialize();
while True:
	trainModel();
	#dumpModel()
	saveModel()
	learningRate = float(input('Learning rate : '));
	for l in net:
		l.learningRate = learningRate
	patienceThreshold = int(input('Patience : '));

