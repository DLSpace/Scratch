import numpy as np
import theano as th
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from theano import function
import time
import math

import unittest
import logging

logging.basicConfig(level=logging.WARNING)


class Layer:
	"""Class that defines a layer in the nnet"""
	neuronCount = 2;
	inputNeuronCount = 0;
	sharedBias = False;
	#activations = [];	#activations for this layer
	#weights = None;
	#biases = None;
	sharedRandomStream = RandomStreams(seed=int(round(time.time())));
	expected = T.fscalar('e');
	learningRate = 0.5;
	#ia = T.dvector('ia');
	#w = T.dmatrix('w');
	#b = T.dvector('b');
	#a = T.nnet.sigmoid((T.dot(ia,w))+b);
	#calcActivationsFunc = function([ia,w,b],a);

	def initializeState(self, kwargs):
		"""
			Initalizer expects three constructor params.
			neuroncount			-	Number of neurons in this layer
			inputNeuronCount	-	Number of neurons in the preceeding layer, zero if this is the input layer
			sharedBias			-	Wether to share a single bias value for the whole layer or to have each neuron to 
			import layer
			from layer import Layer
			il = Layer(neurons=2,inputNeurons=0)
			l1 = Layer(neurons=2,inputNeurons=2)
			ol = Layer(neurons=2,inputNeurons=2)
		"""
		logging.info(kwargs);
		if kwargs.get('neurons') is not None:
			logging.info("neuron count is %s" % (kwargs['neurons']));
			self.neuronCount = int(kwargs['neurons']);
			self.activations = T.zeros(shape=[self.neuronCount]);
		if kwargs.get('inputNeurons') is not None:
			logging.info("inputNeurons count is %s" % (kwargs['inputNeurons']));
			self.inputNeuronCount = int(kwargs['inputNeurons']);
		if kwargs.get('sharedBias') is not None:
			logging.info("Shared Bias is %s" % (kwargs['sharedBias']));
			self.sharedBias = bool(kwargs['sharedBias']);
		if kwargs.get('inputLayer') is not None:
			logging.info("input layer is %s" % (kwargs['inputLayer']));
			self.inputLayer = kwargs['inputLayer'];
		if kwargs.get('name') is not None:
			logging.info("name is %s" % (kwargs['name']));
			self.name = kwargs['name'];

	def randomizeWeightsandBiases(self):
		"""
			n = neurons
			in = input layer neuron count
			a = [1 X n] row matrix
			w = [in X n] 2D matrix where weights from one input neron to next layer are in the column
			and weights from all input neurons are in a row
			--
			|  l
			| w
			|  jk
			--
			J is in current layer so the weight arrow is going from k --> j
			b = [1 X n] row matrix

		"""
		logging.info("name : " + self.name)
		#randomizing weights
		if self.inputNeuronCount>0:
			#randWeights = self.sharedRandomStream.normal([self.inputNeuronCount,self.neuronCount]).eval().copy();
			self.weights =  th.shared(value=(self.sharedRandomStream.normal([self.inputNeuronCount,self.neuronCount]).eval().copy()),name='w');
			logging.info("weights intialized")
		if self.sharedBias:
			logging.info("using shared bias.");
			randBiases = self.sharedRandomStream.normal([1]).eval().copy();
			self.biases = th.shared(value=randBiases,name='b');
		else:
			randBiases = self.sharedRandomStream.normal([self.neuronCount]).eval().copy();
			self.biases = th.shared(value=randBiases,name='b');
		logging.info("biases intialized")

	def __init__(self, **kwargs):
		self.initializeState(kwargs)
		self.randomizeWeightsandBiases()
		self.calculateActivations()
		return

	def calculateActivations(self):
		if self.inputNeuronCount>0:
			self.activations =  T.nnet.sigmoid(T.dot(self.inputLayer.activations,self.weights)+self.biases);
		return self.activations;

	def cost(self):
		return T.sqr(T.sub(self.activations, self.expected)).sum();

	def sgd(self):
		self.gw = T.grad(cost=self.cost(),wrt=self.weights);
		self.gb = T.grad(cost=self.cost(),wrt=self.biases);
		#TODO : parameterize learning rate
		self.updates = [	(self.weights,self.weights - float(self.learningRate) *  self.gw),
					(self.biases,self.biases - float(self.learningRate) * self.gb)
			 ];
		self.sgd_func = th.function(
			inputs = [self.expected],
			outputs = self.cost(),
			updates = self.updates
			#givens = {expected : exp}
			);
		#print(gw.eval())
		#print(gb.eval())
		return self.sgd_func


class LayerTest(unittest.TestCase):
	def test_init(self):
		l1 = Layer(neurons=4,inputNeurons=4,sharedBias=False)
		self.assertEqual(l1.neuronCount,4)
		self.assertEqual(l1.inputNeuronCount,4)
		self.assertFalse(l1.sharedBias)
		self.assertEqual(np.shape(l1.biases),(l1.neuronCount,))
		self.assertEqual(np.shape(l1.weights),(l1.neuronCount,l1.inputNeuronCount))

	def test_layer_zero(self):
		l1 = Layer(neurons=4,inputNeurons=0,sharedBias=False)
		self.assertEqual(l1.neuronCount,4)
		self.assertEqual(l1.inputNeuronCount,0)
		self.assertFalse(l1.sharedBias)
		self.assertEqual(np.shape(l1.biases),(l1.neuronCount,))
		self.assertEqual(np.shape(l1.weights),(l1.inputNeuronCount,))


if __name__ == '__main__':
    unittest.main()
