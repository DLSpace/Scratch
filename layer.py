import numpy as np
import theano as th
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from theano import function
import time

import unittest
import logging

logging.basicConfig(level=logging.WARNING)


class Layer:
	"""Class that defines a layer in the nnet"""
	neuronCount = 2;
	inputNeuronCount = 0;
	sharedBias = False;
	activations = [];	#activations for this layer
	weights = [];
	biases = [];
	sharedRandomStream = RandomStreams(seed=int(round(time.time())));
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


	def randomizeWeightsandBiases(self):
		#randomizing weights
		if self.inputNeuronCount>0:
			randWeights = self.sharedRandomStream.normal([self.inputNeuronCount,self.neuronCount]).eval().copy();
			self.weights =  th.shared(value=randWeights,name='w');
		if self.sharedBias:
			logging.info("using shared bias.");
			randBiases = self.sharedRandomStream.normal([1]).eval().copy();
			self.biases = th.shared(value=randBiases,name='b');
		else:
			randBiases = self.sharedRandomStream.normal([self.neuronCount]).eval().copy();
			self.biases = th.shared(value=randBiases,name='b');

	def __init__(self, **kwargs):
		self.initializeState(kwargs)
		self.randomizeWeightsandBiases()
		return

	def getNeuronCount(self):
		return self.neuronCount;

	def calculateActivations(self, layer):
		if self.inputNeuronCount==0:
			raise Exception("This method is not supported for turminal layer")
		try:
			if not isinstance(layer ,Layer):
				raise TypeError;
			inputActivations = layer.getActivations();
			self.activations =  T.nnet.sigmoid(T.dot(inputActivations,self.weights)+self.biases);
			#self.activations =  calcActivationsFunc(inputActivations,self.weights,self.biases);
		except:
			logging.critical("error happened while calculating activations.")

	def getActivations(self):
		return self.activations;

	def cost(self, expected):
		return T.sqr(T.sub(self.activations, expected)).sum();

	def sgd(self, expected):
		gw = T.grad(cost=self.cost(expected),wrt=self.weights);
		print(gw.eval())
		gb = T.grad(cost=self.cost(expected),wrt=self.biases);
		print(gb.eval())


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
