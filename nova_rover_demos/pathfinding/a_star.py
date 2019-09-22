import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
try:
    from utils.priority_queue import PriorityQueue
except:
    raise
    
from pathfinding.heuristic import manhattan_cost
from math import sqrt, inf
from itertools import product
import numpy as np

def _reconstruct_path_to_destination(prev, end):
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

# def _get_successors(node, grid):
#     """
#     The neighbors of a cell (node) in the grid are the 8-surrounding cells.
#     """
#     successors = []

#     node_x, node_y = node
#     n_rows = len(grid)
#     n_cols = len(grid[0])

#     for dx, dy in product([-1,0,1],[-1,0,1]):
#         # skip the current node itself
#         if (dx == 0 and dy == 0):
#             continue

#         x = node_x + dx
#         y = node_y + dy

#         if (0 <= x < n_rows and 0 <= y < n_cols):
#             cost = grid[y][x]
#         else:
#             # put infinite penalty on successors that would take us off the edge of the grid
#             cost = inf

#         successors.append( ((x, y), cost) )

#     return successors

def _get_neighbors(pos, grid):
    n_cols = len(grid[0])
    n_rows = len(grid)

    neighbors = []
    node_x, node_y = pos
    for dx in [-1, 0, + 1]:
        for dy in [-1, 0, +1]:
            x = node_x + dx
            y = node_y + dy
            if (0 <= x < n_rows) and (0 <= y < n_cols) and (x, y) != pos: 
                neighbors.append((x, y))
    return neighbors

def _node_with_min_fscore(open_set, f_cost): # open_set is a set (of cell) and f_cost is a dict (with cells as keys)
    """
    Find the cell in open set with the smallest f score.
    """
    f_cost_open = dict([a for a in f_cost.items() if a[0] in open_set])
    return min(f_cost_open, key=f_cost_open.get)

def _grid_cost(pos, grid):
    return grid[pos[0]][pos[1]]

def a_star_search(grid, start, end, heuristic_cost=manhattan_cost):
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
    open_set = PriorityQueue()
    open_set.put(start, (heuristic_cost(start, end), _grid_cost(start, grid)))

    # for each cell, mapping to its least-cost incoming cell
    prev = {}

    # for each node, cost of reaching it from start (g_cost)
    # for each node, cost of getting from start to dest via that node (f_cost)
    #   note: cell->dest component of f_cost will be estimated using a heuristic
    g_cost = {}
    for r in range(len(grid)):
        for c in range(len(grid[0])):
            cell = (r, c)
            g_cost[cell] = inf
            
    g_cost[start] = 0
    while not open_set.empty():
        # node in open set with min fscore
        curr = open_set.get()

        # if we've reached the destination
        if curr == end:
            return _reconstruct_path_to_destination(prev, curr)

        for neighbor in _get_neighbors(curr, grid):
            # ignore neighbors which have already been evaluated
            if neighbor in closed_set:
                continue

            curr_g_score =  g_cost[curr] + _grid_cost(neighbor, grid)
            # add neighbor to newly discovered nodes
            if neighbor not in open_set:
                f_cost = g_cost[neighbor] + heuristic_cost(neighbor, end)
                open_set.put(neighbor, (f_cost, _grid_cost(neighbor, grid)))

            # if we've already got a lower g_score for neighbor, then move on
            elif curr_g_score >= g_cost[neighbor]:
                continue

            prev[neighbor] = curr
            g_cost[neighbor] = curr_g_score
        
        closed_set.add(curr)


    # if we get to this point, it's not possible to reach the end destination
    return []
