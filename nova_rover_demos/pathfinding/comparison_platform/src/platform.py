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

'''
    The core function which combines different components.
    It serves as a platform for the data to be passed around between different 
    sub-modules. 

    @param algorithm_list - List of algorithms it needs to compare 
    @param density - The density of obstacles in the environment 
'''


def compare_algorithms(algorithm_list, density):

    maze_x_dimension, maze_y_dimension = 150, 150

    # Generate the arguments to the benchmarking function
    # args output > occupancy_grid, start_coordinates, end_coordinates
    args = [*random_maze(maze_x_dimension, maze_y_dimension, density)]

    # Run the benchmarking on the selected arguments
    time_stats, memory_stats, path_stats = benchmarker(algorithm_list, args)

    # Save  the results in the results folder
    plot_diagram(algorithm_list, args, maze_x_dimension, maze_y_dimension)

    # Print the results in a log file
    print_results(time_stats, memory_stats, path_stats)

    # Draw visuals based on our benchmarking
    visualiser(time_stats, memory_stats, path_stats)
