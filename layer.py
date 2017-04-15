import numpy as np
import theano as th
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import time


class Layer:
	"""Class that defines a layer in the nnet"""
	neuronCount = 2;
	sharedBias = True;
	neurons = [];
	biases = [];
	sharedRandomStream = RandomStreams(seed=int(round(time.time())));

	def __init__(self, **kwargs):
		print(kwargs);
		if kwargs.get('neurons') is not None:
			print("neuron count is %s" % (kwargs['neurons']))
			self.neuronCount = int(kwargs['neurons'])
		self.neurons = self.sharedRandomStream.normal([self.neuronCount]).eval().copy()
		if self.sharedBias:
			self.biases = self.sharedRandomStream.normal([1]).eval().copy()
		else:
			self.biases = self.sharedRandomStream.normal([self.neuronCount]).eval().copy()
		return

	def getNeuronCount(self):
		return self.neuronCount;

	def applySigmoid(self):
		self.neurons = T.nnet.sigmoid(self.neurons).eval();

	def getNeurons(self):
		return self.neurons;
