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