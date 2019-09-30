'''
cloud.py 
Implements class cloud - which is a n dimensional point ploud stored in a numpy array
Author: Jack McRobbie
'''
import numpy as np
class cloud:
	def __init__(self,dims,N,cloud = []):
		self.count = N 
		self.dimension = dims
		if type(cloud) == type([]):
			self.cloudData = cloud
		elif type(cloud) == type(np.array()):
			self.cloudData = np.array([])
	def setCloud(self,data):
		self.cloudData = data