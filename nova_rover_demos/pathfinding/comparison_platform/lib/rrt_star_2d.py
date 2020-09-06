
import numpy as np

from pathfinding.rrtStarCombo.rrt.rrt_star import RRTStar
from pathfinding.rrtStarCombo.usearch_space.search_space import SearchSpace
from pathfinding.rrtStarCombo.utilities.plotting import Plot


# wrapper function
def rrt_star_2d_search(obstacle_list, start, end):
    # instantiate attributes 
    Obstacles = obstacle_list
    
    # dimensions of Search Space
    X_dimensions = np.array([(0, 150), (0, 150)])

    # length of tree edges
    Q = np.array([(8, 4)]) 
 
    # res is taken from fastslam 
    r = res 
    
    # max number of samples to take before timing out
    max_samples = 1024  

    # optional, number of nearby branches to rewire
    rewire_count = 32  
    
    # probability of checking for a connection to goal 
    prc = 0.1   
    
    # create Search Space
    X = SearchSpace(X_dimensions, obstacle_list)

    # create rrt_search
    rrt = RRTStar(X, Q, x_init, x_goal, max_samples, r, prc, rewire_count) 

    try:
        #call the main algorithm
        #it should return the path as a list or coordinates 
        path = rrt.rrt_star()
        maze_solved = True
    except:
        path = start
        maze_solved = False

    # Return a list of tuples as path and
    # a boolean indicating maze solved or not
    return path, maze_solved

"""
Obstacles = ([(x, y, x+res, y+res) #res is taken from fast slam
x_init = (0, 0)  # starting location
x_goal = (100, 100)  # goal location

plot = Plot("rrt_star_2d")
plot.plot_tree(X, rrt.trees)
if path is not None:
    plot.plot_path(X, path)
plot.plot_obstacles(X, Obstacles)
plot.plot_start(X, x_init)
plot.plot_goal(X, x_goal)
plot.draw(auto_open=True) 
"""
