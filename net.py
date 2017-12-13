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
import os, os.path
import glob;

class Net(object):

	def __init__(self, **kwargs):
		self.avgCost = None;
		self.SGDS = [];
		self.net = [];
		self.layerCount = 1;
		self.layerNeuronCount=10;
		self.showPlot = False;
		self.patienceThreshold = 3;
		return super().__init__(**kwargs)

	def copy(self, srcANN):
		self.avgCost = srcANN.avgCost;
		self.SGDS = srcANN.SGDS;
		self.net = srcANN.net;
		self.layerCount = srcANN.layerCount;
		self.layerNeuronCount = srcANN.layerNeuronCount;
		self.showPlot = srcANN.showPlot;
		self.patienceThreshold = srcANN.patienceThreshold;
		self.computeExpressions()
		return self;

	def loadModel(fileName):
		print('loading existing model from ',fileName)	
		model = open(fileName,'rb');
		ANN = pickle.load(model);
		model.close();
		ANNClone = Net();
		ANNClone.copy(ANN);
		#ANNClone.computeExpressions();
		return ANNClone;

	def computeExpressions(self):
		#compute expressions
		self.getInputLayer().activations = theano.shared(value=np.array([np.float32(0)]),name='a',allow_downcast=True);
		for l in self.net:
			l.calculateActivations()
		for l in self.net:
			if hasattr(l,'weights'):
				self.SGDS.extend([l.sgd()])

	def printNet(self):
		print('-------------------------------');
		print('AVG Cost \t: ', self.avgCost);
		print('Layer \t: ', self.layerCount);
		print('Neurons \t: ', self.layerNeuronCount);
		print('Patience \t: ', self.patienceThreshold);
		for l in self.net:
			if hasattr(l,'name'):
				print('Layer name ',l.name)
			if hasattr(l,'weights'):
				print('Weights :-')
				print(l.weights.eval())
			if hasattr(l,'biases'):
				print('Biases :-')
				print(l.biases.eval())
		print('-------------------------------');

	def createNet(self,layerCount,layerNeuronCount):
		self.layerCount = layerCount;
		self.layerNeuronCount = layerNeuronCount;
		self.initialize();

	def initialize(self):
		print('intializing net.')
		il = Layer(neurons=1,inputNeurons=0,name='il');
		self.net.extend([il]);
		for i in range(1,self.layerCount+1,1):
			pl = self.net[i-1]; # previous layer
			l1 = Layer(neurons=self.layerNeuronCount,inputNeurons=pl.neuronCount,inputLayer = pl,name='l'+str(i));
			self.net.extend([l1])
		ol = Layer(neurons=1,inputNeurons=self.net[-1].neuronCount,inputLayer = self.net[-1],name='ol');
		self.net.extend([ol]);
		self.computeExpressions();

	def __makeFileName(self):
		return 'model-L'+str(self.layerCount)+'-N'+str(self.layerNeuronCount)+'-R'+str(self.net[0].learningRate)+'-P'+str(self.patienceThreshold)+'-'+str(self.avgCost);
	
	def saveModel(self):
		fileName = self.__makeFileName() +'.zip';
		self.__saveModel(fileName);
		return fileName;

	def __saveModel(self,fileName):
		model = open(fileName,'wb');
		pickle.dump(self,model,pickle.HIGHEST_PROTOCOL);
		model.close()

	def __saveTemp(self):
		fileName = self.__makeFileName()+'.zip';
		if(not hasattr(self,'tempDir')):
			self.tempDir = self.__makeFileName() + '_temp';
		if(not os.path.exists(self.tempDir)):
			os.mkdir(self.tempDir);
		self.__saveModel(self.tempDir+os.path.sep+ fileName);
		#clean up
		files = glob.glob(self.tempDir+os.path.sep+'*.zip')
		files.sort(key=os.path.getctime);
		if len(files) > 10 :
			for i in range(0,10):
				os.remove(files[i])

	def setLearningRate(self,learningRate):
		for l in self.net:
			l.learningRate = learningRate;
		self.computeExpressions()

	def getLearningRate(self):
		return self.net[0].learningRate;

	def getInputLayer(self):
		return self.net[0];

	def getOutputLayer(self):
		return self.net[len(self.net)-1];

	def train(self):
		epoch=0
		curcost=[]
		fig, ax = plt.subplots()
		plt.ylabel('cost');
		plt.ioff()
		if self.showPlot:
			plt.ion()
			plt.autoscale(enable=True,axis='both');
			plt.show();
		prvCost = 0.0;
		patience = 0;
		inputValsArray = [np.arange(start=0,stop=45,step=2),
						  #np.arange(start=46,stop=90,step=2),
			   			  #np.arange(start=91,stop=135,step=2)
			   ];
		while(patience <= self.patienceThreshold):
			for inputVals in inputValsArray:
				np.random.shuffle(inputVals);
				for ang in inputVals:
					rads = np.float32(math.radians(ang));
					expected = np.float32(math.sin(rads));
					self.getInputLayer().activations.set_value(np.array([rads]));
					#########back propogation##########
					curcost.extend([self.SGDS[-1](expected).tolist()]);
					for i in range(0, -len(self.net),-1):#skips output layer at zero index
						self.SGDS[i](expected)
					###################################
				self.avgCost = (sum(curcost)/len(curcost));
				prctCost = (((prvCost - self.avgCost)/self.avgCost)*100);
				print(self.avgCost,' epoch : ', epoch , '  learning rate: ', self.getLearningRate(),'  change% :' ,prctCost, 'patience : ' , patience);
				prvCost = self.avgCost
				if self.showPlot:
					ax.plot(epoch,self.avgCost,'.');
					plt.pause(0.0001);
					plt.draw();
				if prctCost<=0.0001 :
					#as the learning rate decays adjust the learning rate down for fine tuning
					self.setLearningRate(self.getLearningRate() - (self.getLearningRate() * 0.01));
					patience = patience + 1
				else:
					patience = 0;	
				curcost.clear();
				epoch = epoch+1
				if (epoch % 500)==0 :
					self.__saveTemp()
		plt.savefig(self.__makeFileName() + ".png")
		plt.close()

	def predict(self,ang):
		rads = math.radians(ang);
		self.getInputLayer().activations.set_value(np.array([np.float32(rads)]));
		return self.getOutputLayer().activations.eval()[0];
