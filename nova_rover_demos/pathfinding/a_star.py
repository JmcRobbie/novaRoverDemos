import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
try:
    from utils.priority_queue import PriorityQueue
except:
    raise

from pathfinding.heuristic import euclidean_cost
from math import sqrt, inf
from itertools import product
import numpy as np

def reconstruct_path_to_destination(prev, end):
    """
    Constructs an in-order sequence of (x,y) coordinates (list of tuples)
    to the end destination using the mapping from nodes to their predecessors
    (prev).
    """
    path = [end]
    curr = end
    while curr in prev.keys():
        curr = prev[curr]
        path.insert(0, curr)
    return path

# A* Search
def get_successors(node, grid):
    """
    The neighbors of a cell (node) in the grid are the 8-surrounding cells.
    """
    successors = []

    node_x, node_y = node
    n_rows = len(grid)
    n_cols = len(grid[0])

    for dx, dy in product([-1,0,1],[-1,0,1]):
        # skip the current node itself
        if (dx == 0 and dy == 0):
            continue

        x = node_x + dx
        y = node_y + dy

        if (0 <= x < n_rows and 0 <= y < n_cols):
            cost = grid[y][x]
        else:
            # put infinite penalty on successors that would take us off the edge of the grid
            cost = inf

        successors.append( ((x, y), cost) )

    return successors

def node_with_min_fscore(open_set, f_cost): # open_set is a set (of cell) and f_cost is a dict (with cells as keys)
    """
    Find the cell in open set with the smallest f score.
    """
    f_cost_open = dict([a for a in f_cost.items() if a[0] in open_set])
    return min(f_cost_open, key=f_cost_open.get)

def a_star_search(grid, start, end, heuristic_cost=euclidean_cost):
    """
    Implementation of A Star over a 2D grid. Returns a list of waypoints
    as a list of (x,y) tuples.
    Input:
    : grid, 2D matrix
    : start, (x,y) tuple, start position
    : end, (x,y) tuple, end destination
    Output:
    : waypoints, list of (x,y) tuples
    """
    # the set of cells already evaluated
    closed_set = set()

    # the set of cells already discovered
    open_set = set()
    open_set.add(start)

    # for each cell, mapping to its least-cost incoming cell
    prev = {}

    # for each node, cost of reaching it from start (g_cost)
    # for each node, cost of getting from start to dest via that node (f_cost)
    #   note: cell->dest component of f_cost will be estimated using a heuristic
    g_cost = {}
    f_cost = {}
    for r in range(len(grid)):
        for c in range(len(grid[0])):
            cell = (r, c)
            g_cost[cell] = inf
            f_cost[cell] = inf
    g_cost[start] = 0
    f_cost[start] = heuristic_cost(start, end)

    while open_set:
        # node in open set with min fscore
        curr = node_with_min_fscore(open_set, f_cost)

        # if we've reached the destination
        if curr == end:
            return reconstruct_path_to_destination(prev, curr)

        open_set.remove(curr)
        closed_set.add(curr)

        for neighbor, cost in get_successors(curr, grid):
            # ignore neighbors which have already been evaluated
            if neighbor in closed_set:
                continue

            curr_g_score =  g_cost[curr] + cost
            # add neighbor to newly discovered nodes
            if neighbor not in open_set:
                open_set.add(neighbor)

            # if we've already got a lower g_score for neighbor, then move on
            elif curr_g_score >= g_cost[neighbor]:
                continue

            prev[neighbor] = curr
            g_cost[neighbor] = curr_g_score
            f_cost[neighbor] = g_cost[neighbor] + heuristic_cost(neighbor, end)

    # if we get to this point, it's not possible to reach the end destination
    return []
<<<<<<< master:nova_rover_demos/pathfinding/a_star.py
=======

def null_heuristic(curr, end):
    return 0

def dijkstra(grid,start,end, heuristic_cost=null_heuristic):
    """
    Implementation of Dijkstra Star over a 2D grid. Returns a list of waypoints
    as a list of (x,y) tuples. This exploits the special case of a_star reducing
    to dijkstra's algorithm when the heuristic is a null heuristic.

    Input:
    : grid, 2D matrix
    : start, (x,y) tuple, start position
    : end, (x,y) tuple, end destination

    Output:
    : waypoints, list of (x,y) tuples
    """
    return a_star(grid,start,end, heuristic_cost)


class PriorityQueue:
    """
    Queue data structure that returns the highest priority item in the queue
    """
    def __init__(self):
        self._elements = []

    """
    Method to return boolean value indicating if PriorityQueue is empty
    """
    def empty(self):
        return len(self._elements) == 0

    """
    Add element item to queue 
    """
    def put(self, item, priority):
        heapq.heappush(self._elements, (priority, item))

    """
    Method to return highest priority item in queue 
    @param state : current state in board 
    @return      : highest priority item in queue if start is none, else the item with the same state 
    """
    def get(self, state=None):
        return heapq.heappop(self._elements)[1]

    """
    Return length of queue
    """
    def len(self):
        return len(self._elements)

    """
    Method to return boolean value indicating if item is contained in the queue 
    """
    def __contains__(self, other):
        for item in self._elements:
            if other == item[1]:
                return True
        return False

    """
    Method to return the index of an item if it occurs in the queue 
    """
    def index(self, item):
        for index, elem in enumerate(self._elements):
            if item == elem[1]:
                return index
        return -1


# Greedy Search
def is_obstacle(pos, grid, threshold=0.75):
    return grid_cost(pos, grid) > threshold

def grid_cost(pos, grid):
    return grid[pos[0]][pos[1]]

def is_goal(pos, goal):
    return pos == goal

def reconstruct_path(end_pos, prev):
    result = [end_pos]

    curr_pos = end_pos 
    while curr_pos in prev.keys():
        curr_pos = prev[curr_pos]
        result.insert(0, curr_pos)
    return result

def get_neighbours(pos, grid):
    n_cols = len(grid[0])
    n_rows = len(grid)

    neighbours = []

    for i in [-1, 0, + 1]:
        for j in [-1, 0, +1]:
            x = pos[0] + i
            y = pos[1] + j 
            if (0 <= x < n_rows) and (0 <= y < n_cols) and (x, y) != pos: 
                neighbours.append((x, y))
    return neighbours

def greedy_search(grid, start, end, heuristic_cost=manhattan_heuristic_cost):
    closed_set = set()
    open_set = PriorityQueue()

    open_set.put(start, (heuristic_cost(start, end), grid_cost(start, grid)))

    prev = {}

    while not open_set.empty():
        curr = open_set.get()

        if is_goal(curr, end):
            return reconstruct_path(curr, prev)

        for neighbour in get_neighbours(curr, grid):
            if neighbour in closed_set:
                continue

            elif is_obstacle(neighbour, grid):
                continue

            if neighbour not in open_set:
                open_set.put(neighbour, (heuristic_cost(neighbour, end), grid_cost(neighbour, grid)))

            prev[neighbour] = curr
        closed_set.add(curr)

    return []

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

def plot_grid(grid, path):
    plt.imshow(grid, cmap='hot', interpolation='nearest')

    x_pos = []
    y_pos = []

    for x in path:
        x_pos.append(x[1])
        y_pos.append(x[0])
    plt.plot(x_pos,y_pos, "ro",color = "Green")

    plt.show()
    
def test_search(grid, search):
    start =  (1, 1)
    goal = (len(grid) - 1, len(grid[0]) - 1)
    path = search(grid, start, goal)
    print(path)
    plot_grid(grid, path)

    

# test_grid = norm_grid()

# print("ASTAR")
# test_search(test_grid, a_star)
# print("Greedy")
# test_search(test_grid, greedy_search)
# test_search(test_grid, dijkstra)
>>>>>>> Added Graph-SLAM implementation:pathfinding.py
