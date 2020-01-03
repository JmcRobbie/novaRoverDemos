'''
Model predictive control for a skid steering robot. 

Pyhsical model adapted from the following paper(s), code my own.

- http://yadda.icm.edu.pl/yadda/element/bwmeta1.element.baztech-500d8121-3587-4ee9-9800-7ddc3626b8c4

Author: Jack McRobbie - Melbourne Space Program 

'''
import numpy as np
import math
import cvxpy

class roverState: 
    X = np.zeros([1,3])
    def __init__(self,state):
        '''
        state vector X = [x y theta v]^T
        '''
        self.X[0] = state[0]
        self.X[1] = state[1]
        self.X[2] = state[2]
        self.X[3] = state[3]

    def update(self, acc, omega, dt):
        '''
        Update the state of the rover, takes the control inputs and time step as inputs. 
        '''


    def motionModel(self): 
        ''' 
        Implements the motion model for the rover
        ''' 

        A = np.zeros([4,4])
        
        return A

def pi_2_pi(angle):
    while(angle > math.pi):
        angle = angle - 2.0 * math.pi

    while(angle < -math.pi):
        angle = angle + 2.0 * math.pi

    return angle
