import numpy as np
import random

import matplotlib.pyplot as plt

def random_obstacle(x, y, max_size):
    obstacle_size = random.randint(5,max_size)
    obstacle_occupancy = [(x,y)]
    for i in range(obstacle_size):
        x += random.choice([1, 0, -1])
        y += random.choice([1, 0, -1])
        obstacle_occupancy.append((x,y))
    return obstacle_occupancy



def random_maze(x_dimension, y_dimension, density):
    oc_grid = []
    if density == 'heavy':
        num_obstacles = int( np.sqrt(x_dimension * y_dimension) )
        max_obstacle_size = int( np.sqrt(x_dimension * y_dimension) )
    elif density == 'medium':
        num_obstacles = int( 0.75 * np.sqrt(x_dimension * y_dimension) )
        max_obstacle_size = int( np.sqrt(x_dimension * y_dimension) )
    elif density == 'light':
        num_obstacles = int( 0.5 * np.sqrt(x_dimension * y_dimension) )
        max_obstacle_size = int( np.sqrt(x_dimension * y_dimension) )
    elif density == 'sparse':
        num_obstacles = int( 0.25 * np.sqrt(x_dimension * y_dimension) )
        max_obstacle_size = int( np.sqrt(x_dimension * y_dimension) )

    start = (0,0)
    end = (x_dimension - 1, y_dimension - 1)

    for i in range(num_obstacles):
        x = random.randint(1, x_dimension - 2)
        y = random.randint(1, y_dimension - 2)
        for i in random_obstacle(x,y, max_obstacle_size):
            if 0 <= i[0] <= x_dimension - 2 and 0 <= i[1] <= y_dimension - 2:
                oc_grid.append(i)
    '''
    Start and End positions are either in the corner or centre of the edge of the maze
    Start and End are always on opposite edges of the maze
    '''
    waypoints = [0, int(x_dimension/2), x_dimension - 1]
    start = (random.choice(waypoints), 0)
    if start[0] != int(x_dimension/2): #Prevent the maze from generating start and end coordinates along the edge of the maze
        del waypoints[waypoints.index(start[0])]

    end = (random.choice(waypoints), y_dimension - 1)

    return oc_grid, start, end
