from src.a_star import PriorityQueue

# Dijkstra's Search implementation 
def dijkstra_search(graph, start, goal):
    frontier = PriorityQueue()
    frontier.put(start, 0)
    # A dictionary to track the source of nodes 
    came_from = {}
    came_from[start] = None 
    # Dictionary to maintain the cost of travelling to that node 
    travel_cost = {}
    travel_cost[start] = 0

    while not frontier.isEmpty():
        current = frontier.get()

        # Check if goal has been found or not 
        if current == goal:
            break
        
        # Check the neighbors of the current node 
        for next in graph.neighbors(current):
            new_cost = travel_cost[current] + graph.cost(current, next)

            # Check if node have not been travelled before or we have found the 
            # less costly path to that node 
            if next not in travel_cost or new_cost < travel_cost[next]:
                travel_cost[next] = new_cost
                # Put the item in the priority queue 
                priority = new_cost
                frontier.put(next, priority)
                came_from[next] = current

    return came_from, travel_cost



# Implementation of BFS 
import collections

# This queue class is just a wrapper around deque class 
class Queue: 
    def __init__(self):
        self.elements = collections.deque(); 

    def isEmpty(self):
        return len(self.elements) == 0

    def put(self, x):
        self.elements.append(x)
    
    def get(self):
        return self.elements.popleft()

'''
    Parameters: 
        graph - The graph it will be searching on and of WeightedGrid class 
        start - Starting coordinate in form (x, y)
        goal - Target coordinate in form (x, y)
'''
def bfs_search(graph, start, goal):
    # We will print what we find 
    frontier = Queue()
    frontier.put(start)
    # Dictionary to hold the visited nodes and their source 
    came_from = {}
    came_from[start] = None

    # Perfrom the breadth first search 
    while not frontier.isEmpty():
        current = frontier.get()

        # Check if goal is reached or not 
        if current == goal: 
            break 
        # Check for neighbors 
        for next in graph.neighbors(current): 
            if next not in came_from:
                frontier.put(next)
                came_from[next] = current
    
    return came_from
