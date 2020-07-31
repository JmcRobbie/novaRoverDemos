'''
Current version of 'modified pledge algorithm' designed to find a path from a start to end position

Notes / Problems:
 - Inefficient on very large mazes: uses if '{x} in list' to check whether a given cell is a wall: causes large time delays for mazez with many walls
 - Is not a perfect algorithm, ie. there are mazes where it cannot find the solution and gets stuck in infinite loops
'''
from labyrinth import Labyrinth
from mouse import Mouse

def modified_pledge(occupancy_grid, start_position, end_position):
    maze = Labyrinth(occupancy_grid, start_position, end_position)
    rover = Mouse(maze)
    while rover.has_reached_exit() == False:
        rover.move()
    return rover.path, rover.infinite_loop
