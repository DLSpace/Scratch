import numpy
import theano
import theano.tensor as T
from theano import shared
from theano import function
from theano import pp
import layer
from layer  import Layer
import math

il = Layer(neurons=1,inputNeurons=0);
l1 = Layer(neurons=2,inputNeurons=1);
l2 = Layer(neurons=2,inputNeurons=2);
ol = Layer(neurons=1,inputNeurons=2);

il.activations = numpy.array([math.radians(30)])
l1.calculateActivations(il)
l2.calculateActivations(l1)
ol.calculateActivations(l2)
expected = math.sin(math.radians(30))
print('Expected output is %f and actual output is %f' % (expected, ol.activations.eval()))
print('Cost function %s' % (ol.cost(expected).eval()))
ol.sgd(expected)


#training


