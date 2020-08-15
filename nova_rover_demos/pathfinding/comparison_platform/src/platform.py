import time
import random
import numpy as np

from src.maze.random_maze import *
from src.benchmark import *
from src.visuals import *
'''
Syntax: random_maze(x_dimension, y_dimension, density)
    > Density can be 'heavy', 'medium', 'light', or 'sparse'
'''


def compare_algorithms(algorithm_list, density):
    
    maze_x_dimension, maze_y_dimension = 40, 40

    # Generate the arguments to the benchmarking function
    # args output > occupancy_grid, start_coordinates, end_coordinates
    args = [*random_maze(maze_x_dimension, maze_y_dimension, density)]
    
    # Run the benchmarking on the selected arguments 
    time_stats, memory_stats, path_stats = benchmarker( algorithm_list, args )

    # Draw visuals based on our benchmarking 
    visualiser(time_stats, memory_stats, path_stats)