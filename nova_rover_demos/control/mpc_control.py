'''
Model predictive control for a skid steering robot. 

Pyhsical model adapted from the following paper(s), code my own.

- http://yadda.icm.edu.pl/yadda/element/bwmeta1.element.baztech-500d8121-3587-4ee9-9800-7ddc3626b8c4
Author: Jack McRobbie - Melbourne Space Program 

'''
import numpy as np
class roverState: 
    self.X = np.zeros([1,3])
    def __init__(self,state):
        '''
        state vector X = [x y theta]^T
        '''
        self.X[0] = state[0]
        self.X[1] = state[1]
        self.X[2] = state[2]
state_test = np.array([1, 2, 3])
print state_test