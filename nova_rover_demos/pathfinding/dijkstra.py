from pathfinding.heuristic import null_cost
from pathfinding.a_star import a_star_search

def dijkstra_search(grid,start,end, heuristic_cost=null_cost):
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
    return a_star_search(grid,start,end, heuristic_cost)