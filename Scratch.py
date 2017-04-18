import numpy
import theano
import theano.tensor as T
from theano import shared
from theano import function
from theano import pp
import layer
from layer  import Layer

il = Layer(neurons=1,inputNeurons=0);
l1 = Layer(neurons=2,inputNeurons=1);
l2 = Layer(neurons=2,inputNeurons=2);
ol = Layer(neurons=1,inputNeurons=2);

l1.calculateActivations(il)
print(l1.activations)

#training


