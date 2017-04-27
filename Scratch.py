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



il = Layer(neurons=1,inputNeurons=0,name='il');
l1 = Layer(neurons=2,inputNeurons=1,inputLayer = il,name='l1');
l2 = Layer(neurons=2,inputNeurons=2,inputLayer = l1,name='l2');
ol = Layer(neurons=1,inputNeurons=2,inputLayer = l2,name='ol');

il.activations = theano.shared(value=np.array([np.float32(0)]),name='a',allow_downcast=True);
il.calculateActivations();
l1.calculateActivations();
l2.calculateActivations();
ol.calculateActivations();
l1SGD = l1.sgd();
l2SGD = l2.sgd();
olSGD = ol.sgd();


#training
inputVals = np.arange(start=0,stop=90,step=2);
counter=0
curcost=[]

plt.ion()
fig, ax = plt.subplots()
plt.ylabel('cost');
plt.show();
ax.set_ylim([0,1]);
ax.set_xlim([0,25]);
plt.autoscale(enable=True,axis='both');
prvCost = 0.9;
patience = 0;
while(patience <= 150):
	for ang in inputVals:
		rads = np.float32(math.radians(ang));
		expected = np.float32(math.sin(rads));
		il.activations.set_value(np.array([rads]));
		curcost.extend([olSGD(expected).tolist()])
		l2SGD(np.float32(ol.activations.eval().mean()));
		l1SGD(np.float32(l2.activations.eval().mean()));
	avgCost = (sum(curcost)/len(curcost));
	prctCost = (((prvCost - avgCost)/avgCost)*100);
	print(avgCost, '  change% :' ,prctCost, 'patience : ' , patience);
	prvCost = avgCost
	ax.plot(counter,avgCost,'.');
	plt.pause(0.0001);
	plt.draw();
	if prctCost<=0.001 :
		patience = patience + 1
	else:
		patience = 0;	
	curcost.clear();
	counter = counter+1

#testing
def predict(ang):
	rads = math.radians(ang);
	il.activations.set_value(np.array([np.float32(rads)]));
	print(ol.activations.eval(), math.sin(rads));

inputVals = np.arange(start=91,stop=180,step=10);
for ang in inputVals:
	predict(ang);

ang = int(input('Angle : '));
#while(ang>=0):
#	predict(ang);
#	ang = int(input('Angle : '));


