import numpy as np

class Labyrinth:

    def __init__(self, walls, start, end):
        self.walls = walls
        self.walls_dict = self.read_walls_to_dict()

        self.start_pos = tuple(start)
        self.goal = tuple(end)

        self.directions = [0, 1, 2, 3]
        self.movements = np.array([ [1,0], [0,1], [-1,0], [0,-1] ])

    '''
    Directions Structure:
    | Right = 0 | up = 1 | left = 2 | down = 3 |
    '''
    def get_allowed_directions(self, position):
        mouse_position = np.array(position)

        allowed_directions = []
        for direction, movement in zip(self.directions, self.movements):
            if not self.is_wall(mouse_position + movement):
                allowed_directions.append(direction)
        return allowed_directions

    def is_wall(self, position):
        out = False
        if tuple(position) in self.walls_dict:
            out = True
        return out
        #Do this with a set of tuples or dictionary - constant time lookup

    def read_walls_to_dict(self):
        walls_dict = {}
        for i in self.walls:
            walls_dict[i] = 0
        return walls_dict
