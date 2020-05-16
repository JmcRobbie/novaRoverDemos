import numpy as np
import math
from matplotlib import pyplot
import pickle

bins = [[1,2], [2,3]]



def writeCloudToFile(fileName, points):
    with open(fileName + '.pts', 'wb') as filehandle:
        pickle.dump(points, filehandle)

def readCloudToFile(fileName):
    points = []
    
    with open(fileName + '.pts', 'rb') as filehandle:
        points = pickle.load(filehandle)
            
    return points
            