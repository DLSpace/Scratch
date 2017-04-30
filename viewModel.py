import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-f", "--fileName", help="number of layers in the net")
args = parser.parse_args()

import six.moves.cPickle as pickle
import net
from net import Net
import PyQt5 as qt

print('loading existing model.')	
model = open(fileName,'rb');
ANN = pickle.load(model);
model.close()
ANN.printNet()

