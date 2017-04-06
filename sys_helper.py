import os;
import importlib;


def sys(cmd):
	os.system(cmd)

def cls():
	sys('cls')

def cd():
	sys('cd')

def pwd():
	cd()

	
def reload(module):
	importlib.reload(module)

