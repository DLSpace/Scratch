import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-f", "--fileName", help="number of layers in the net")
parser.add_argument("-s", "--start", help="Start angle",type=int)
parser.add_argument("-e", "--end", help="end angle",type=int)
args = parser.parse_args()
import numpy as np
#import theano
#import theano.tensor as T
#from theano import shared
#from theano import function
#import theano.printing as pp
#import layer
from layer  import Layer
import math
#import sys
#import matplotlib.pyplot as plt
import six.moves.cPickle as pickle


fileName = args.fileName
startAngle = args.start
endAngle = args.end

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

def predict(ang):
	rads = math.radians(ang);
	il.activations.set_value(np.array([np.float32(rads)]));
	print(ol.activations.eval()[0], ',', math.sin(rads));

print('loading existing model.')	
model = open(fileName,'rb');
net = pickle.load(model);
model.close()
dumpModel()
il = net[0];
ol = net[len(net)-1];
for ang in range(startAngle, endAngle,1):
	predict(ang)
