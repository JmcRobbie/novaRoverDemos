import numpy as np
import matplotlib.pyplot as plt
import time

class Mouse:

    def __init__(self, labyrinth):
        self.__labyrinth = labyrinth
        self.position = labyrinth.start_pos

        self.direction = self.direction_to_goal() #Number between 0-3

        self.wall_following = False
        self.wall_following_direction = 'Left'
        self.sum_turns = 0

        self.path = [self.position]
        self.path_dict = {}
        self.path_dict[self.position] = 1

        self.infinite_loop = False

    def direction_to_goal(self):
        x,y = np.array(self.__labyrinth.goal) - np.array(self.position)
        theta = np.arctan(y/x) * 180/np.pi
        '''
        Account for ambiguity from arctan: specify direction around unit circle
        '''
        if x < 0:
            if y >= 0:
                theta += 180
            else:
                theta -= 180
        directions = np.array([0, 90, 180, -90])
        idx = np.abs(directions - theta).argmin()
        return idx

    def has_reached_exit(self):
        if self.check_loop(): #Checks for loops to end the code]
            self.infinite_loop = True
            out = True
        else:
            out = tuple(self.position) == self.__labyrinth.goal
        return out

    def turn_right(self):
        return (self.direction - 1)%4

    def turn_left(self):
        return (self.direction + 1)%4

    def next_right(self):
        dir = self.direction
        for j in range(1,3+1):
            if (dir - j)%4 in self.__labyrinth.get_allowed_directions(self.position):
                out = (dir - j)%4
                break
        return out, -j

    def next_left(self):
        dir = self.direction
        for j in range(1,3+1):
            if (dir + j)%4 in self.__labyrinth.get_allowed_directions(self.position):
                out = (dir + j)%4
                break
        return out, j

    def continue_straight(self):
        return self.direction, 0

    def step(self, turn_direction):
        return tuple( np.array(self.position) + np.array(self.__labyrinth.movements[turn_direction]) )

    def check_loop(self):
        '''
        Returns 'True' if a single cell is visited for a third time (Less visits might be sufficient to detect loop)
        '''
        return max(self.path_dict.values()) >= 15

    def move(self):
        if self.has_reached_exit() == False:
            '''
            Logic for moving directly towards goal
            '''
            if self.wall_following == False:
                if self.direction_to_goal() in self.__labyrinth.get_allowed_directions(self.position):
                    self.position = tuple(self.position + self.__labyrinth.movements[self.direction_to_goal()])
                    self.direction = self.direction_to_goal()
                    self.path.append(tuple(self.position))
                else:
                    self.wall_following = True                   #If a wall is encountered, begin wall following

                    if self.path.count(self.position) > 1:   #Turn right if location has been visited before
                        self.wall_following_direction = 'Right'
                        '''
                        Begin wall following phase by first aligning mouse perpendicular to wall
                        '''
                        self.direction, num_turns = self.next_right()
                        self.sum_turns += num_turns

                    else:
                        self.wall_following_direction = 'Left'
                        '''
                        Begin wall following phase by first aligning mouse perpendicular to wall
                        '''
                        self.direction, num_turns = self.next_left()
                        self.sum_turns += num_turns
            else:
                '''
                Logic for Wall Following
                '''
                if self.wall_following_direction == 'Left':
                    if self.turn_right() in self.__labyrinth.get_allowed_directions(self.position):
                        '''
                        Turn Right if Possible
                        '''
                        self.direction, num_turns = self.next_right()
                        self.sum_turns += num_turns
                        self.position = tuple(self.step(self.direction))

                    elif self.direction in self.__labyrinth.get_allowed_directions(self.position):
                        '''
                        Continue Straight if no right turn is available
                        '''
                        self.position = tuple(self.step(self.direction))

                    else:
                        '''
                        Take the next left if no other option available
                        '''
                        self.direction, num_turns = self.next_left()
                        self.sum_turns += num_turns
                        self.position = tuple(self.step(self.direction))

                elif self.wall_following_direction == 'Right':
                    if self.turn_left() in self.__labyrinth.get_allowed_directions(self.position):
                        '''
                        Turn Right if Possible
                        '''
                        self.direction, num_turns = self.next_left()
                        self.sum_turns += num_turns
                        self.position = tuple(self.step(self.direction))

                    elif self.direction in self.__labyrinth.get_allowed_directions(self.position):
                        '''
                        Continue Straight if no right turn is available
                        '''
                        self.position = tuple(self.step(self.direction))

                    else:
                        '''
                        Take the next right if no other option available
                        '''
                        self.direction, num_turns = self.next_right()
                        self.sum_turns += num_turns
                        self.position = tuple(self.step(self.direction))

                self.path.append(tuple(self.position)) #Update ordered path for output
                '''
                Update the path dictionary for loop detection
                '''
                try: self.path_dict[tuple(self.position)] += 1
                except: self.path_dict[tuple(self.position)] = 1

                if self.sum_turns == 0 or abs(self.sum_turns) >= 5:
                    self.wall_following = False
                    self.sum_terms = 0
