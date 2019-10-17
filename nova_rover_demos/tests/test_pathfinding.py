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
    
def search(grid, search, title, plot_grid_enable=False):
    start =  (1, 1)
    goal = (len(grid) - 1, len(grid[0]) - 1)
    path = search(grid, start, goal)

    if plot_grid_enable:
        plot_grid(grid, path, title)

    return get_path_cost(grid, path)

def test_a_star_search():
    a = search(norm_grid(), a_star_search, "A-Star")
    assert 5 < a < 6

def test_best_first_search():
    a = search(norm_grid(), best_first_search, "Best-First")
    assert 7 < a < 8

def test_dijkstra_search():
    a = search(norm_grid(), dijkstra_search, "Dijkstra")
    assert 10 < a < 11

def test_max_filter_grid_a_star_search():
    a = search(maximum_filter(norm_grid(), size=(3,3)), a_star_search, "A-Star")
    assert 12 < a < 13

def test_max_filter_grid_best_first_search():
    a = search(maximum_filter(norm_grid(), size=(3,3)), best_first_search, "Best-First")
    assert 32 < a < 33

def test_max_filter_grid_dijkstra_search():
    a = search(maximum_filter(norm_grid(), size=(3,3)), dijkstra_search, "Dijkstra")
    assert 13 < a < 14


if __name__ == '__main__':
    search(norm_grid(), a_star_search, "A-Star", plot_grid_enable=True)
    search(norm_grid(), best_first_search, "Best-First", plot_grid_enable=True)
    search(norm_grid(), dijkstra_search, "Dijkstra", plot_grid_enable=True)
    search(maximum_filter(norm_grid(), size=(3,3)), a_star_search, "A-Star", plot_grid_enable=True)
    search(maximum_filter(norm_grid(), size=(3,3)), best_first_search, "Best-First", plot_grid_enable=True)
    search(maximum_filter(norm_grid(), size=(3,3)), dijkstra_search, "Dijkstra", plot_grid_enable=True)