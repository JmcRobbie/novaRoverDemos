'''
Implementation of the fast slam algorithm 
For a step by step implementation please check the Jupyter Notebook
NOTE: 

We do not own the entire code 
The code is partly taken from the Python Robotics Repo 
'''

import math

import matplotlib.pyplot as plt
import numpy as np

# Fast SLAM covariance
Q = np.diag([3.0, np.deg2rad(10.0)]) ** 2
R = np.diag([1.0, np.deg2rad(20.0)]) ** 2

#  Simulation parameter
Qsim = np.diag([0.3, np.deg2rad(2.0)]) ** 2
Rsim = np.diag([0.5, np.deg2rad(10.0)]) ** 2
OFFSET_YAWRATE_NOISE = 0.01

DT = 0.1  # time tick [s]
SIM_TIME = 50.0  # simulation time [s]
MAX_RANGE = 20.0  # maximum observation range
M_DIST_TH = 2.0  # Threshold of Mahalanobis distance for data association.
STATE_SIZE = 3  # State size [x,y,yaw]
LM_SIZE = 2  # LM srate size [x,y]
N_PARTICLE = 100  # number of particle
NTH = N_PARTICLE / 1.5  # Number of particle for re-sampling

show_animation = True
