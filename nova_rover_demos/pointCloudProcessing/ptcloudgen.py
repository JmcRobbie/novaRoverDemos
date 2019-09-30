'''
ptcloudgen.py 

A class that generates dummy point clouds in accordance with user configurations. 
The point clouds are designed to be mappable to a surface in order to be coherently 
processable by processes.

Author: Jack McRobbie
'''
import math
import numpy as np
import random as rn
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D

class point_cloud_gen:

	def __init__ (self,size, features,noise = 1,N = 10000):
		'''
		size: the bounds on the range of the point cloud values. Is a list of n tuples where n is the number of dimensions
		eg. [(-1,1),(-2,5)]
		'''

		self.dims = len(size)
		if self.dims == 0 or self.dims > 3:
			raise Exception("Input length of size should not be 0 or greater than 3")
		self.var = 1
		self.num_features = features
		self.num_points = N
		self.size = size
	def _gen_white_noise(self):
		''' 
		Generates an output cloud of points with a uniform random distribution.
		No specific feautures.
		'''
		if self.num_features is not 0:
			raise Exception("Uniform cloud cannot have more than 0 features")
		dat = np.zeros([self.num_points,self.dims])
		for i in range(self.num_points):
			for j in range(self.dims):
				dat[i][j] = rn.uniform(self.size[j][0],self.size[j][1])
		self.ptcloud = dat
		return
	def uniformCloud(self):
		'''
		Public implementation of white noise generator
		'''
		self.num_features = 0 ## Encorced by cloud type.
		self._gen_white_noise()
		return 

	def plotCloud(self):
		fig = pyplot.figure()
		ax = Axes3D(fig)
		x = self.ptcloud[:,0]
		y = self.ptcloud[:,1]
		z = self.ptcloud[:,2]
		ax.scatter(x,y,z)
		pyplot.show()
	def GetOccupancyGrid(self,x_res,y_res):
		'''
		Returns an occupancy grid of the point cloud over the relevant range.
		Occupancy is computed based on a heuristic that can be configured.  
		'''
		xSteps = int(abs(self.size[0][0] -self.size[0][1])/x_res)
		ySteps = int(abs(self.size[1][0] -self.size[1][1])/y_res)
		obstacleCloud = self.ptcloud[np.where(self.ptcloud[:,2]>0.2)]
		self.occupancy = np.zeros([xSteps,ySteps]) ## construct occupancy grid
		for point in obstacleCloud:
			xc = int(point[0]/x_res)
			yc = int(point[1]/y_res)
			self.occupancy[xc][yc] = self.occupancy[xc][yc] + point[2]
			if self.occupancy[xc][yc]>10.0:
				 self.occupancy[xc][yc] = 10.0
		return self.occupancy

	def plotOccGrid(self):
		'''
		Plots a heat map of the occupancy grid for the class.
		'''
		pyplot.imshow(self.occupancy, cmap='hot', interpolation='nearest')
		pyplot.show()

	def _gaussian(self,center,x,y,weight = 1.0, sigma = (1.0,1.0)):
		'''
		Returns a 3d gaussian functions value at point x,y with the given parameters
		'''

		return math.e**(-(x-center[0])**2/(2*sigma[0]**2) -(y-center[1])**2/(2*sigma[1]**2 ))*weight 

	def gaussianCloud(self):
		''' 
		Generates a series of gaussian peaks which are then observed by the point clouds 
		Number of peaks is defined by the number of features. 
		'''
		gaussians = []
		if self.num_features is 0:
			self.ptcloud = dat
			return

		dat = np.zeros([self.num_points,self.dims])
		for i in range(self.num_features):
			place = (self.size[0][0],self.size[0][1],self.size[1][0],self.size[1][1])
			gaussians.append(place)
		
		for i in range(self.num_points):
			dat[i][0] = rn.uniform(self.size[0][0],self.size[0][1])
			dat[i][1] = rn.uniform(self.size[1][0],self.size[1][1])

			dat[i][2] = 0
			for j in range(self.num_features):
				dat[i][2] = dat[i][2] + self._gaussian((0.5,0.5), dat[i][0],dat[i][1],1.0,(0.1,0.1))
		self.ptcloud = dat
		return
	
def test_occGrid():
	size = [(0,1),(0,2),(0,5)]
	cld = point_cloud_gen(size,1)
	cld.gaussianCloud()
	cld.GetOccupancyGrid(0.05,0.05)
	cld.plotOccGrid()
	cld.plotCloud()

def test_ptcloud():
	size = [(0,1),(0,2),(0,5)]
	cld = point_cloud_gen(size,0)
	cld.uniformCloud()
	cld.plotCloud()

if __name__ == '__main__':
	test_occGrid()