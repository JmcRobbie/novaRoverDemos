import time 

# Importing a-star and family  
from src.a_star import * 
from src.a_star_variants import *
from src.other_algorithms import *

# Importing necessary diagrams and grid visualizer 
from maze.diagrams import *
from maze.grid import draw_grid

grid = diagram3

# #Testing a-star 
start, goal = (1, 4), (38, 28)
# Calculating the cost and node origins 
print('-------------------- Regular A* -----------------------\n')
path = a_star_search(grid, start, goal)
draw_grid(grid, width=3, path=path)
print('\n')


# Testing variants  
print('-------------------- Bidirectional A* -----------------------\n')
path = bidirectional_a_star(grid, start, goal)
draw_grid(grid, width=3, path=path)
print('\n')


print('-------------------- Weighted A* -----------------------\n')
path = weighted_a_star(grid, start, goal)
draw_grid(grid, width=3, path=path)
print('\n')
