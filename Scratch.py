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

il = Layer(neurons=1,inputNeurons=0,name='il');
l1 = Layer(neurons=2,inputNeurons=1,inputLayer = il,name='l1');
l2 = Layer(neurons=2,inputNeurons=2,inputLayer = l1,name='l2');
ol = Layer(neurons=1,inputNeurons=2,inputLayer = l2,name='ol');
olSGD = ol.sgd();

inputVals = [0,30,45,60,90];
#fh = open('c:\\temp\\test.txt','wb');
#training
counter=0
curcost=[]
while(counter <= 100):
	for ang in inputVals:
		rads = math.radians(ang);
		sinVal = math.sin(rads);
		il.activations = theano.shared(value=np.array([rads]),name='a');
		expected = np.float32(sinVal);
		curcost.extend([olSGD(expected)])
	avgCost = (sum(curcost)/len(curcost));
	print(avgCost)
	if avgCost<=0.0001 :
		break;
	curcost.clear();
	counter = counter+1

rads = 33
il.activations = theano.shared(value=np.array([rads]),name='a');
print(ol.activations.eval(), math.sin(math.radians(rads)));


