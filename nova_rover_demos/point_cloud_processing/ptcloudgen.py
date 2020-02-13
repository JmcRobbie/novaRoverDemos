"""
ptcloudgen.py

A class that generates dummy point clouds in accordance with user configurations.
The point clouds are designed to be mappable to a surface in order to be coherently
processable by processes.

Author: Jack McRobbie and Xinpu Cheng
"""

# Changelog: 
'''
1. Added comments detailing the instance variables of class 'PointCloudGen'.
2. Removed the intermediary variable 'cloud' from the method 'from_numpy_array'.
3. Removed the intermediary variable 'dat' from the methods '_gen_white_noise' and 'gaussian_cloud'.
4. Delegated the task of generating xSteps and ySteps from the methods 'get_occupancy_grid', 'meanHeightGenerator' and 'gradientComputation' to a helper function 'gen_2D_steps'.
'''

import math
import numpy as np
import random as rn
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D


class PointCloudGen:
    '''
    This class contains the following instance variables:
    size: The bounds on the range of the point cloud values. It's a list of n tuples where n is the number of dimensions,
        eg. [(-1,1),(-2,5)],
    ptcloud: A list of lists of the points in x, y, z coordinates,
    occupancy: Features/Obstacles represented as x, y coordinates (Any value that is non-zero is considered a(n) feature/obstacle),
    series: A series of features/obstacles represented as a list of tuples with grid resolution taken into consideration,
    dims: The number of dimensions involved,
    num_features,
    num_points.
    '''
    
    def __init__(self, size, num_features, noise=1, num_points=10000, pt=[]): 
        '''
        NOTE: All units and measurements are in SI units.
        '''
        self.ptcloud = pt
        
        self.dims = len(size)
        if self.dims == 0 or self.dims > 3:
            raise Exception("Input length of size should not be equal to 0 or greater than 3")
            
        self.var = 1
        self.num_features = num_features
        self.num_points = num_points
        self.size = size

    @classmethod
    def from_numpy_array(self, ptcloud):
        '''
        Construct the class from a numpy array.
        '''
        dims = len(ptcloud)
        var = 1
        num_features = -1
        num_points = len(ptcloud[0])
        noise = 1
        x_range = (min(ptcloud[:, 0]), max(ptcloud[:, 0]))
        y_range = (min(ptcloud[:, 1]), max(ptcloud[:, 1]))
        z_range = (min(ptcloud[:, 2]), max(ptcloud[:, 2]))
        size = [x_range, y_range, z_range]

        return self(size, num_features, noise, num_points, ptcloud)

    def _gen_white_noise(self):
        '''
        Generates an output cloud of points with a uniform random distribution.
        No specific feautures.
        '''
        if self.num_features is not 0:
            raise Exception("Uniform cloud cannot have more than 0 features")

        self.ptcloud = np.zeros([self.num_points, self.dims])
        for i in range(self.num_points):
            for j in range(self.dims):
                self.ptcloud[i][j] = rn.uniform(self.size[j][0], self.size[j][1])

        return

    def uniform_cloud(self):
        '''
        Public implementation of white noise generator
        '''
        self.num_features = 0  # Encorced by cloud type.
        self._gen_white_noise()
        
        return

    def plot_cloud(self):
        fig = pyplot.figure()
        ax = Axes3D(fig)
        x = self.ptcloud[:, 0]
        y = self.ptcloud[:, 1]
        z = self.ptcloud[:, 2]
        ax.scatter(x, y, z)
        pyplot.show()

    def _gen_2D_steps(self, x_res, y_res):
        x_lower = self.size[0][0]
        x_upper = self.size[0][1]
        y_lower = self.size[1][0]
        y_upper = self.size[1][1]
        xSteps = int(abs(x_lower - x_upper) / x_res)
        ySteps = int(abs(y_lower - y_upper) / y_res)
        
        return (xSteps, ySteps)
    
    def get_occupancy_grid(self, x_res, y_res, get_occupancy_in_series=False):
        '''
        Returns an occupancy grid of the point cloud over the relevant range.
        Occupancy is computed based on a heuristic that can be configured.
        '''
        xSteps, ySteps = self._gen_2D_steps(x_res, y_res)
        # Anything that's above 20 cm is considered an obstacle
        obstacleCloud = self.ptcloud[np.where(self.ptcloud[:, 2] > 0.2)]
        self.occupancy = np.zeros([xSteps, ySteps])  # Construct occupancy grid
        
        for point in obstacleCloud:
            xc = int(point[0] / x_res)
            yc = int(point[1] / y_res)
            self.occupancy[xc][yc] = self.occupancy[xc][yc] + point[2]
            # 10.0 is a maximum threshold/limit
            if self.occupancy[xc][yc] > 10.0:
                self.occupancy[xc][yc] = 10.0

        if get_occupancy_in_series:
            self.series = self._get_occupancy_in_series(x_res, y_res)

        return self.occupancy

    def _get_occupancy_in_series(self, x_res, y_res):
        '''
        Convert the occupancy grid into a series of points, i.e., a list of tuples of x, y coordinates
        Requirement: self.occupancy (Occupancy grid representation)
        Note: The rover's relative position is assumed to be at point (size[0][0], size[1][0])
        '''
        x_shift = self.size[0][0]
        y_shift = self.size[1][0]
        result = []
        x_pos = 0

        for row_data in self.occupancy:
            y_pos = 0
            for col_value in row_data:
                if col_value != 0:
                    result.append((x_pos * x_res + x_shift, y_pos * y_res + y_shift))
                y_pos += 1
            x_pos += 1

        return result

    def meanHeightGenerator(self, x_res, y_res):
        xSteps, ySteps = self._gen_2D_steps(x_res, y_res)
        self.average = np.zeros([xSteps, ySteps])  # Construct occupancy grid
        count_grid = np.zeros([xSteps, ySteps])
        sum_grid = np.zeros([xSteps, ySteps])
        
        for point in self.ptcloud:
            xc = int(point[0] / x_res)
            yc = int(point[1] / y_res)
            count_grid[xc][yc] += 1
            sum_grid [xc][yc] = point[2]
            
        for i in range(xSteps):
            for j in range(ySteps):
                self.average[i][j] = sum_grid[i][j]/count_grid[i][j]
                
    def gradientComputation(self, x_res, y_res):
        '''
        Computes a gradient for the mean height estimation.
        Can then be used to evaluate for traversibility via some heuristic.
        '''
        xSteps, ySteps = self._gen_2D_steps(x_res, y_res)
        self.gradient = np.zeros([xSteps, ySteps])
        
        for i in range(xSteps):
            for j in range(ySteps):
                if i == 0 or j ==0 or i == xSteps-1 or j == ySteps-1:
                    self.gradient[i][j] = 0
                else:
                    delx = (self.average[i-1][j] - self.average[i+1][j])/x_res
                    dely = (self.average[i][j-1] - self.average[i][j+1])/y_res
                    grad = abs(delx)+abs(dely)
                    self.gradient[i][j] = grad
                    
    def plot_gradient_grid(self):
        '''
        Plots a heat map of the gradient map
        '''
        pyplot.imshow(self.gradient, cmap = 'hot', interpolation = 'nearest')
        pyplot.show()
        
    def plot_occupancy_grid(self):
        '''
        Plots a heat map of the occupancy grid for the class.
        '''
        pyplot.imshow(self.occupancy, cmap='hot', interpolation='nearest')
        pyplot.show()

    def _gaussian(self, center, x, y, weight=1.0, sigma=(1.0, 1.0)):
        '''
        Returns a 3d gaussian functions value at point x,y with the given parameters
        '''
        return math.e**(-(x - center[0])**2 / (2 * sigma[0]**2) -
                        (y - center[1])**2 / (2 * sigma[1]**2)) * weight

    def gaussian_cloud(self):
        '''
        Generates a series of gaussian peaks which are then observed by the point clouds
        Number of peaks is defined by the number of features.
        '''
        gaussians = []
        if self.num_features is 0:
            return

        self.ptcloud = np.zeros([self.num_points, self.dims])
        for i in range(self.num_features):
            place = (
                rn.uniform(
                    self.size[0][0],
                    self.size[0][1]),
                rn.uniform(
                    self.size[1][0],
                    self.size[1][1]))
            gaussians.append(place)

        for i in range(self.num_points):
            self.ptcloud[i][0] = rn.uniform(self.size[0][0], self.size[0][1])
            self.ptcloud[i][1] = rn.uniform(self.size[1][0], self.size[1][1])

            self.ptcloud[i][2] = 0
            for j in range(self.num_features):
                self.ptcloud[i][2] = self.ptcloud[i][2] + \
                    self._gaussian(gaussians[j], self.ptcloud[i][0], self.ptcloud[i][1], 1.0, (0.1, 0.1))
                
        return


# Unit tests
def test_occupancy_grid():
    size = [(0, 1), (0, 2), (0, 5)]
    cld = PointCloudGen(size, 1)
    cld.gaussian_cloud()
    cld.get_occupancy_grid(0.05, 0.05, True)
    print("Ptcloud coordinates:", cld.ptcloud)
    print("Occupancy grid coordinates:", cld.occupancy)
    print("Occupancy length:", cld.occupancy.shape)
    print("Ptcloud length:", cld.ptcloud.shape)
    print("Series of points:", cld.series)
    cld.plot_occupancy_grid()
    cld.plot_cloud()


def test_ptcloud():
    size = [(0, 1), (0, 2), (0, 5)]
    cld = PointCloudGen(size, 0)
    cld.uniform_cloud()
    cld.plot_cloud()


def test_numpy_constructor():
    arr = np.random.rand(100, 3)
    ptcl = PointCloudGen.from_numpy_array(arr)
    print(ptcl.ptcloud)
    

if __name__ == '__main__':
    test_occupancy_grid()
    test_ptcloud()
    test_numpy_constructor()
