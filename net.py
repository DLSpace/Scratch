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


def setupNet():
	il = Layer(neurons=1,inputNeurons=0,name='il');
	l1 = Layer(neurons=2,inputNeurons=1,inputLayer = il,name='l1');
	l2 = Layer(neurons=2,inputNeurons=2,inputLayer = l1,name='l2');
	ol = Layer(neurons=1,inputNeurons=2,inputLayer = l2,name='ol');
	olSGD = ol.sgd();
	return il,l1,l2,ol,olSGD;
