from math import sqrt, inf
from itertools import product
import matplotlib.pyplot as plt
import heapq
import numpy as np
import random
from utils.priority_queue import PriorityQueue
from heuristic import manhattan_cost

# A* Search

# Greedy Search
def _is_obstacle(pos, grid, threshold=0.75):
    return grid_cost(pos, grid) > threshold

def _grid_cost(pos, grid):
    return grid[pos[0]][pos[1]]

def _is_goal(pos, goal):
    return pos == goal

def _reconstruct_path(end, prev):
    result = [end]

    curr = end 
    while curr in prev.keys():
        curr = prev[curr]
        result.insert(0, curr)
    return result

def _get_neighbours(pos, grid):
    n_cols = len(grid[0])
    n_rows = len(grid)

    neighbours = []
    node_x, node_y = pos
    for dx in [-1, 0, + 1]:
        for dy in [-1, 0, +1]:
            x = node_x + dx
            y = node_y + dy
            if (0 <= x < n_rows) and (0 <= y < n_cols) and (x, y) != pos: 
                neighbours.append((x, y))
    return neighbours

def greedy_search(grid, start, end, heuristic_cost=manhattan_cost):
    closed_set = set()
    open_set = PriorityQueue()

    open_set.put(start, (heuristic_cost(start, end), grid_cost(start, grid)))

    prev = {}

    while not open_set.empty():
        curr = open_set.get()

        if is_goal(curr, end):
            return _reconstruct_path(curr, prev)

        for neighbour in _get_neighbours(curr, grid):
            if neighbour in closed_set:
                continue

            elif _is_obstacle(neighbour, grid):
                continue

            if neighbour not in open_set:
                open_set.put(neighbour, (heuristic_cost(neighbour, end),
                                         grid_cost(neighbour, grid)))

            prev[neighbour] = curr
        closed_set.add(curr)

    return []