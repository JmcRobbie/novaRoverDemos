from modified_pledge import *
from demo_labyrinth import * #Imports the occupancy grid of a 150x150 maze as a list cells that contain walls, as tuples


import numpy as np
import matplotlib.pyplot as plt
import time

#__..__--__..__--__..__..__--__..__--__..#
#__..__--__..__--__..__..__--__..__--__..#

def show_path(oc_grid, path, start, end):

    min_x = abs( min( min([i[0] for i in oc_grid]) , min([i[0] for i in path]) ) )
    min_y = abs( min( min([i[1] for i in oc_grid]) , min([i[1] for i in path]) ) )


    oc_grid = [tuple(np.add(np.array(i),[min_x, min_y])) for i in oc_grid]
    path = [tuple(np.add(np.array(i),[min_x, min_y])) for i in path]


    max_x = max( max([i[0] for i in oc_grid]) , max([i[0] for i in path]), start[0], end[0] ) + 1
    max_y = max( max([i[1] for i in oc_grid]) , max([i[1] for i in path]), start[1], end[1] ) + 1

    grid = np.ones([max_x, max_y])

    for j in oc_grid:
        grid[j] = 0
    for k in path:
        grid[k] = 2

    grid[start] = 3
    grid[end] = 4

    plt.imshow(grid.T)
    plt.colorbar()
    plt.show()

def png_to_occupancy(mazefile):
    '''
    Takes 4 channel png image, converts to occupancy grid where 0's are walls
    '''
    grid = plt.imread(mazefile).dot([1,0,0,0])

    wall_y, wall_x = np.where(grid == 0)
    return [tuple([wall_x[i], wall_y[i]]) for i in range(len(wall_x))]

#__..__--__..__--__..__..__--__..__--__..#
#__..__--__..__--__..__..__--__..__--__..#

'''
Labyrinth is 'oc_grid': a list of tuples each containing the coordinates of a 'wall' cell, imported from 'demo_labyrinth.py'
Arbitrary png files can be converted into this format using the 'png_to_occupancy()'' function above.
'''

start_position = tuple([2,2])
end_position = tuple([150, 150])

#<<<#####>>>#
start_time = time.time()
path, loop_status = modified_pledge(oc_grid, start_position, end_position)
end_time = time.time()
#<<<#####>>>#

print(f'\n\nAlgorithm Solved Maze? > {not loop_status}\n\nRuntime = {end_time - start_time:.2f} Seconds')
show_path(oc_grid, path, start_position, end_position)
