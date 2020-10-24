''' 
This file is provided for additional testing of the different algorithms
'''

from new_maze.generate_maze import maze_generator
import time

# Importing a-star and family
from src.a_star import *
from src.a_star_variants import *
from src.other_algorithms import *

# Importing necessary diagrams and grid visualizer
from maze.diagrams import *
from maze.grid import draw_grid
from new_maze.random_maze import *

# Defining start and goal
start, goal = (1, 4), (38, 28)
# # Get the desired path
# path = a_star_search(grid1, start, goal)
# # Print the length of path
# print(len(path))

# # Trying a dimension specific map
# grid2 = diagram3
# path = weighted_a_star(grid2, start, goal)
# # Print the length of path
# print(len(path))

maze_parameters = maze_generator(10, 10, 'light')
# print(*maze_arguments)
path = a_star_search(*maze_parameters)
print(path)
