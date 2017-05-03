import signal,os
import threading

def signalReceived(signum,frame):
	print('Signal received')

signal.signal(signal.CTRL_BREAK_EVENT,signalReceived)
while(True):
	print('waiting for signal..')
	
