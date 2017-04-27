#import numpy as np
#import theano
#import theano.tensor as T
#from theano import shared
#from theano import function
#import theano.printing as pp
#import layer
from layer  import Layer
#import math
#import sys
#import matplotlib.pyplot as plt
import six.moves.cPickle as pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--fileName", help="number of layers in the net")
args = parser.parse_args()

fileName = args.fileName

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


print('loading existing model.')	
model = open(fileName,'rb');
net = pickle.load(model);
model.close()
dumpModel()
