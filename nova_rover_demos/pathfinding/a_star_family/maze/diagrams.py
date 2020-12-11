from collections import defaultdict
import random

"""
SquareGrid class - map representation  
Holds location tuples (int, int) -> (x, y)

This grid cells does not have weights 
"""


class SquareGrid:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.walls = defaultdict(int)

    # Method to check if location is within the map
    def in_bounds(self, id):
        (x, y) = id
        return 0 <= x < self.width and 0 <= y < self.height

    # Check if current location is blocked for not
    def passable(self, id):
        return False if self.walls[id] == 1 else True

    # Check the neighbors of the current grid
    def neighbors(self, id):
        (x, y) = id
        result = [(x+1, y), (x, y-1), (x-1, y), (x, y+1)]
        # Just for aesthetics
        if(x + y) % 2 == 0:
            result.reverse()
        # Check if the neighbors are in the map and not blocked
        result = list(filter(self.in_bounds, result))
        result = list(filter(self.passable, result))

        return result

    # Helper method to add walls to the dictionary
    def add_walls(self, walls):
        for wall in walls:
            self.walls[wall] = 1


""" 
A Class which also can represent the cost of movement
Extends the SquareGrid class to add extra functionality 

"""


class WeightedGrid(SquareGrid):
    def __init__(self, width, height):
        super().__init__(width, height)
        self.weights = {}

    # Method to get cost to travel from weights, else default value
    def cost(self, from_node, to_node):
        return self.weights.get(to_node, 1)


""" 
A class to represent an open grid which does not have any fixed dimensions. 
We can assume got these kind of grids we will only be given information about 
the obstacles and nothing else. 

This is the class that would be used with Rover as the FastSlam algorithm only
provides us with a list of coordinates of the obstacles 
"""


class OpenGrid:
    def __init__(self):
        self.walls = defaultdict(int)
        self.weights = {}

    # Check if current location is blocked for not
    def passable(self, id):
        return False if self.walls[id] == 1 else True

    # Check the neighbors of the current grid
    def neighbors(self, id):
        (x, y) = id
        result = [(x+1, y), (x, y-1), (x-1, y), (x, y+1)]
        # Just for aesthetics
        if(x + y) % 2 == 0:
            result.reverse()
        # Check if the neighbors are not blocked
        result = list(filter(self.passable, result))

        return result

    # Method to get cost to travel from weights, else default value of 1
    def cost(self, from_node, to_node):
        return self.weights.get(to_node, 1)

    # Helper method to add walls to the dictionary
    def add_walls(self, walls):
        for wall in walls:
            self.walls[wall] = 1


        ########################################### Implementation of diagrams #######################################
        # A 50 by 50 grid with equal grid values
diagram1 = WeightedGrid(50, 50)
walls = [(3, i) for i in range(31)]
walls = walls + [(15, i) for i in range(30, 50)]
walls = walls + [(30, i) for i in range(31)]
walls = walls + [(45, i) for i in range(30, 50)]
diagram1.add_walls(walls)

# Generating a random obstacle grid
diagram2 = WeightedGrid(40, 30)
walls = []
for wall in range(65):
    x = random.randint(3, 35)
    y = random.randint(0, 30)
    walls.append((x, y))

diagram2.add_walls(walls)


# Weighted grid with random values
diagram3 = WeightedGrid(40, 40)
walls = []
for wall in range(65):
    x = random.randint(3, 35)
    y = random.randint(0, 30)
    walls.append((x, y))

diagram3.add_walls(walls)

for x in range(40):
    for y in range(40):
        val = random.randint(1, 6)
        diagram3.weights[(x, y)] = val


# Implementation for weighted grid
diagram4 = WeightedGrid(40, 40)
walls = [(1, 7), (1, 8), (2, 7), (2, 8), (3, 7), (3, 8)]
diagram4.add_walls(walls)
diagram4.weights = {loc: 5 for loc in [(3, 4), (3, 5), (4, 1), (4, 2),
                                       (4, 3), (4, 4), (4, 5), (4, 6),
                                       (4, 7), (4, 8), (5, 1), (5, 2),
                                       (5, 3), (5, 4), (5, 5), (5, 6),
                                       (5, 7), (5, 8), (6, 2), (6, 3),
                                       (6, 4), (6, 5), (6, 6), (6, 7),
                                       (7, 3), (7, 4), (7, 5)]}


# Implementation of the open-grid

diagram5 = OpenGrid()
walls = []
for wall in range(65):
    x = random.randint(3, 35)
    y = random.randint(0, 30)
    walls.append((x, y))

diagram5.add_walls(walls)
