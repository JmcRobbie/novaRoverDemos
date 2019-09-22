from math import sqrt, inf

def euclidean_cost(curr, end):
    """
    Estimates cost from curr (x0,y0) to end (x1,y1) using Euclidean
    distance.
    """
    curr_x, curr_y = curr
    end_x, end_y = end
    return sqrt((curr_x-end_x)**2 + (curr_y-end_y)**2)

def manhattan_cost(curr, end):
    """
    Estimates cost from curr (x0,y0) to end (x1,y1) using Manhattan
    distance.
    """
    curr_x, curr_y = curr
    end_x, end_y = end
    return abs(curr_x-end_x) + abs(curr_y-end_y)

def null_cost(curr, end):
    return 0