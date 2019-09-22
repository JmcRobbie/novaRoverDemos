import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
try:
    from pathfinding.a_star import a_star_search
    from pathfinding.dijkstra import dijkstra_search
    from pathfinding.best_first import best_first_search
except:
    raise
    
import numpy as np
import matplotlib.pyplot as plt
from math import inf
from scipy.ndimage.filters import maximum_filter
    
def basic_grid():
    array = [0.0, 0.1, 0.6, 0.8]
    grid = np.random.choice(array, (8, 8))
    return grid

def norm_grid(size=(15, 15)):
    grid = []
    n_rows, n_cols = size
    for i in range(n_rows):
        grid.append([])

        for j in range(n_cols):
            val = np.random.normal(loc=0.2, scale = 0.7)
            while val < 0 or val > 1:
                val = np.random.normal(loc=0.5, scale=0.7)
            grid[i].append(val)

    return grid

def get_path_cost(grid, path):
    if path is None:
        return inf
    total_cost = 0
    for pos in path:
        x, y = pos
        total_cost += grid[x][y]
    return total_cost

def plot_grid(grid, path, title):
    path_cost = get_path_cost(grid, path)
    plt.imshow(grid, cmap='hot', interpolation='nearest')

    x_pos = []
    y_pos = []

    for x, y in path:
        x_pos.append(x)
        y_pos.append(y)
        
    plt.plot(x_pos,y_pos, "ro",color = "Green")
    plt.title("Path cost: " + str(path_cost), fontsize=10)
    plt.suptitle("Path demonstration of " + title + " Search", y=0.97, fontsize=16)
    plt.show()
    
def test_search(grid, search, title):
    start =  (1, 1)
    goal = (len(grid) - 1, len(grid[0]) - 1)
    path = search(grid, start, goal)
    plot_grid(grid, path, title)

test_grid = norm_grid()

test_search(test_grid, a_star_search, "A-Star")
test_search(test_grid, best_first_search, "Best-First")
test_search(test_grid, dijkstra_search, "Dijkstra")

max_filter_grid = maximum_filter(test_grid, size=(3,3))

test_search(max_filter_grid, a_star_search, "A-Star")
test_search(max_filter_grid, best_first_search, "Best-First")
test_search(max_filter_grid, dijkstra_search, "Dijkstra")

