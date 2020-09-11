from lib.rrt_star.rrt_star_2d import *
from src.maze.random_maze import *

args = [*random_maze(150, 150, 'heavy')]


print(rrt_star_2d_search(*args))
