from maze.diagrams import OpenGrid
from new_maze.random_maze import *


def maze_generator(x, y, density):
    maze_arguments = random_maze(x, y, density)

    diagram = OpenGrid()
    diagram.add_walls(maze_arguments[0])
    start = maze_arguments[1]
    end = maze_arguments[2]

    return diagram, start, end
