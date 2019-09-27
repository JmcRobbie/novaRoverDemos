'''
ptcloudgen.py 

A class that generates dummy point clouds in accordance with user configurations. 
The point clouds are designed to be mappable to a surface in order to be coherently 
processable by processes.

Author: Jack McRobbie
'''

class point_cloud_gen:
	self.dims = NaN
	def __init__ (self,size, features, noise = 1,):
		'''
		size: the bounds on the range of the point cloud values. Is a list of n tuples where n is the number of dimensions
		eg. [(-1,1),(-2,5)]
		'''
		self.dims = len(size)
		if self.dims = 0 or self.dims > 3:
			raise Exception("Input length of size should not be 0 or greater than 3. The vale was: {}" format(self.dim))
		self.var = 1
		self.num_features = features
	def 