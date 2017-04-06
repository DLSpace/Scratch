# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
import json

def readJsonF(fileName):
    """ Opens a file tries to load the json into an object and returns the object """
    fileHandle = open(fileName,'r')
    try:
        data = json.load(fileHandle)
        return data
    except IOError:
        print("IO error")
    finally:
        fileHandle.close()

def writeJsonF(fileName, data):
    """ Opens a file tries to write the data as json into the file"""
    fileHandle = open(fileName,'w')
    try:
        data = json.dump(data,fileHandle)
        return data
    except IOError:
        print("IO error")
    finally:
        fileHandle.close()
