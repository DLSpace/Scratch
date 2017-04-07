## Module to put all the math functions
import numpy as np
import theano as th
import theano.tensor as T
import theano.tensor.nnet as nnet

def test1():
    nnet.sigmoid(np.array([1.3434,3.3434,5.565656,6])).eval()