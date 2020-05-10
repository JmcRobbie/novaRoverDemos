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

    # 
    #  Ground Plane Extraction
    #  @caelana
    # 
    
    def calculateSegment(self, x, y, seg_size):
        # casting to int is faster than rounding (~3x)
        return int(np.arctan2(y,x) / seg_size)
    
    def calculateDistance(self, x, y):
        return math.sqrt(x**2 + y**2)

    def fitLine(self, points):
        '''
        Uses least squares fit\\
        Points must be numpy array
        '''
        a = np.vstack([x, np.ones(len(x))]).T
        return np.dot(np.linalg.inv(np.dot(a.T, a)), np.dot(a.T, y))
        
    def fitError(self, m, b, points):
        predictions = points[:,1]
        targets = [m*x + b for x in points[:,0]]
        return np.sqrt(((predictions - targets) ** 2).mean())
    
    def getPrototypePoint(self,points):
        '''
        Uses heuristic to map set of points to a single one \\
        Best for ground plane extraction is taking the min
        '''
        min_i = np.argmin(points[:,1])
        return points[min_i]
    
    def distpointline(self, point, m, b):
        return abs(point[1] - m*point[0] + b)
    
    def isGroundPoint(self, x, y, line_segments, ground_distance_threshold):
        # Find closest line segment
        # endpoint_pairs = np.array([[m*x+b, ]])
        
        
        # int(np.argmin(a) / 2)
        
        # for l in line_segments:
        #     m = line_segments[0]
        #     b = line_segments[1]
            
            
    
    def extractSegmentLines(self, bins, slope_range, plateu_threshold, rmse_threshold, line_endpoint_threshold):
        '''
        Takes in bins for given segment (along with parameters)\\
        Returns list of line segments described by their gradient and intercept (m,b) for a given angular segment
        Main idea is to model the the distribution of prototype points along the segment bins with the least number of line segments 
        '''
        n_bins = len(segment_bins)
        
        c = 0
        line_segments = np.array([])
        points_to_fit = np.array([])
        
        for i in range(n_bins):
            bin_points = np.array(segment_bins[i])
            
            if bin_points is not None and len(bin_points) > 0:
                if len(segment_points) >= 2:     
                    points_to_fit_with_prototype = points_to_fit + np.array([getPrototypePoint(bin_points)])
                    
                    (m, b) = self.fitLine(points_to_fit_with_prototype)
                    
                    if(abs(m) <= slope_range[1] and (abs(m) > slope_range[1] or abs(b) <= plateu_threshold) \
                         and self.fitError(m,b, points_to_fit_with_prototype) <= rmse_threshold):
                        points_to_fit = points_to_fit_with_prototype
                    else:
                        (m, b) = self.fitLine(points_to_fit)
                        line_segments += [(m,b)]
                        c += 1
                        points_to_fit = 0
                        i -= 1
                else:
                    if(c == 0 or points_to_fit == 0 or self.distpointline(getPrototypePoint(bin_points)) < line_endpoint_threshold):
                        points_to_fit += [getPrototypePoint(bin_points)]
        return  line_segments     
            
                
        

    def groundPlaneComputation(self, n_seg = 10, n_bins = 10, radial_range = [0,3], slope_range = [2, 5], 
                               plateu_threshold = 10, rmse_threshold = 1, line_endpoint_threshold = 10):
        '''
        Calculates ground plane, returning a boolean filled occupancy grid.\\
        Implementation of paper: https://ieeexplore.ieee.org/document/5548059 \\
        Main idea is reducing points from point cloud into a discrete cylindrical space for fast 2D line extraction.
        This allows one to compare the original point cloud with the extracted lines and using thresholds, determine points belonging to ground plane
        
        n_seg: number of angular steps \\
        n_bins: number of radial steps per segment \\
        radial_range: min,max radial values for grouping points into a bin of a given segment \\
        slope_range: m < min => small slope, m > max => obstacle \\
        plateu_threshold: if m < slope_range.min and b < plateu_threshold => edge of plateu \\
        rmse_threshold: used as an error threshold in line fitting  \\  
        line_endpoint_threshold: lines with endpoints on the same bin must not have a difference in endpoint height being greater than this threshold \\
        '''        
        seg_size =  2*np.math.pi / n_seg
        bin_size = np.abs(radial_range[1] - radial_range[0]) / n_bins

        # Group points by segment
        segments = {}
        for i in range(self.num_points):
            segment_i = self.calculateSegment(self.ptcloud[i][0], self.ptcloud[i][1], seg_size)
            if(segments.get(segment_i) is None):
                segments[segment_i] = []
            segments[segment_i] += [self.ptcloud[i]]
        
        # Group points by bin
        bins = {} # first index is segment, second index is bin
        for i,k in enumerate(segments):
            print("Binning in Segment #{}\n".format(k))
            if bins.get(k) is None:
                bins[k] = {}
            for p in segments[k]:
                radial_dist = self.calculateDistance(p[0], p[1])
                bin_i = int(radial_dist / bin_size)
                if(bins[k].get(bin_i) is None):
                    bins[k][bin_i] = []
                bins[k][bin_i] += [radial_dist, p[2]]
        
            # Line Extraction
            segment_lines = self.extractSegmentLines(bins[k], slope_range, plateu_threshold, rmse_threshold, line_endpoint_threshold)


        pass

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

def test_ground_plane_extraction():
    size = [(0, 2), (0, 2), (0, 2)]
    cld = PointCloudGen(size, 1, num_points=100)
    cld.getPrototypePoint(np.array([[2,6], [1,5], [1,7]]))
    # cld.gaussian_cloud()
    # cld.plot_cloud()
    # cld.groundPlaneComputation()

def test_numpy_constructor():
    arr = np.random.rand(100, 3)
    ptcl = PointCloudGen.from_numpy_array(arr)
    print(ptcl.ptcloud)
    

if __name__ == '__main__':
    test_ground_plane_extraction()
    # test_occupancy_grid()
    # # test_ptcloud()
    # # test_numpy_constructor()
