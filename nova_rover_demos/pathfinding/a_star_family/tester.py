''' 
This file is provided for additional testing of the different algorithms
'''

import time 

# Importing a-star and family  
from src.a_star import * 
from src.a_star_variants import *
from src.other_algorithms import *

# Importing necessary diagrams and grid visualizer 
from maze.diagrams import *
from maze.grid import draw_grid

# Trying a open grid system 
grid1 = diagram5

# Defining start and goal  
start, goal = (1, 4), (38, 28)
# Get the desired path 
path = a_star_search(grid1, start, goal)
# Print the length of path 
print(len(path))

# Trying a dimension specific map 
grid2 = diagram3
path = weighted_a_star(grid2, start, goal)
# Print the length of path 
print(len(path))