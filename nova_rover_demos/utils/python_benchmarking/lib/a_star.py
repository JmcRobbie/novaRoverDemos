import math 

# Implementing a priority queue with heaps 
import heapq 

# Wrapper class with added functionalities 
class PriorityQueue:
    def __init__(self):
        self.elements = []
    
    def isEmpty(self):
        return len(self.elements) == 0

    def put(self, item, priority):
        heapq.heappush(self.elements, (priority, item))

    def get(self):
        # Return only the item 
        return heapq.heappop(self.elements)[1]


# Function to reconstruct path from start to goal using the source dictionary 
def reconstruct_path(came_from, start, goal):
    current = goal 
    path = []

    while current != start:
        path.append(current)
        current = came_from[current]

    # Add the start to the path - optional 
    path.append(start)
    # reverse the path to get from start to goal 
    path.reverse()
    return path

# Combine two paths and avoid happing overlapping nodes 
def join_paths(path_1, path_2):
    path_2.reverse() 
    if(path_1[-1] == path_2[0]):
        return path_1 + path_2[1:]


# The heuristic used by the a-star algorithm 
# This is used to calculate the distance of current node form the goal too 
def manhattan_heuristic(a, b):
    (x1, y1) = a
    (x2, y2) = b

    return abs(x1 - x2) + abs(y1 - y2)

# Diagonal heuristic which considers the diagonal value of nodes 
def diagonal_heuristic(a, b):
    (x1, y1) = a
    (x2, y2) = b

    return max(abs(x1 - x2), abs(y1 - y2))

# Taking the euclidian distance as heuristics 
def euclidian_heuristic(a, b):
    (x1, y1) = a
    (x2, y2) = b

    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

# Implementation of the a-star search algorithm 
def a_star_search(graph, start, goal):
    # Priority Queue track progression of nodes 
    frontier = PriorityQueue()
    frontier.put(start, 0)
    
    # Dictionary to track origin of a node 
    came_from = {}
    # Dictionary to track the cost to move to a particular node 
    cost_so_far = {}

    # Add starting node into the dictionaries 
    came_from[start] = None
    cost_so_far[start] = 0 

    # While the Priority queue is not empty 
    while not frontier.isEmpty():
        # Get the top of queue 
        current = frontier.get()

        # check if we have reached destination 
        if current == goal: 
            break 

        # Loop through neighbors of current node and process them 
        for next in graph.neighbors(current):
            # Calculate the new cost to travel to neighboring node 
            # The new cost is cost of travelling to current node plus the cost of
            # travelling from current node to the neighbor 
            new_cost = cost_so_far[current] + graph.cost(current, next)
            
            # Check if this node hasn't been reached before or we have a new cheaper path
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                # Set the priority of the neighbor using the heuristic 
                # We are taking distance to goal in consideration through heuristics 
                priority = new_cost + manhattan_heuristic(goal, next)
                frontier.put(next, priority)
                came_from[next] = current

    
    # Return the cost and source dictionary 
    #return came_from, cost_so_far
    return reconstruct_path(came_from, start, goal)


