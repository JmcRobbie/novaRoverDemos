from src.a_star import *

# Implementation of bidirectional a-start
# This algorithm will start looking for a path from both the start and the goal 
def bidirectional_a_star(graph, start, goal):
    # Priority queues to hold the progression of nodes 
    # This will hold nodes starting from the 'start' node 
    frontier_1 = PriorityQueue()
    frontier_1.put(start, 0)
    # This will node nodes starting from the 'goal' node 
    frontier_2 = PriorityQueue()
    frontier_2.put(goal, 0)

    # Dictionaries to hold the cost of travel and origin of nodes 
    # For the start node 
    came_from_1 = {}
    cost_so_far_1 = {}

    came_from_1[start] = None 
    cost_so_far_1[start] = 0 


    # For the goal node 
    came_from_2 = {}
    cost_so_far_2 = {}

    came_from_2[goal] = None
    cost_so_far_2[goal] = 0 

    # Run the loop until both of the 
    while not frontier_1.isEmpty() or frontier_2.isEmpty():
        # Get the top from each of the queues 
        current_1 = frontier_1.get()
        current_2 = frontier_2.get()

        #Check if we have found a full path or not 
        if current_1 == goal or current_2 == start: 
            break 
        # We have found two overlapping node 
        if current_1 in cost_so_far_2 and current_2 in cost_so_far_1:
            path_1 = reconstruct_path(came_from_1, start, current_1) 
            path_2 = reconstruct_path(came_from_2, goal, current_2)
            combined_path = join_paths(path_1, path_2)
            break
        # Only one of the nodes are are overlapping 
        elif (current_1 in cost_so_far_2):
            path_1 = reconstruct_path(came_from_1, start, current_1) 
            path_2 = reconstruct_path(came_from_2, goal, current_1)
            combined_path = join_paths(path_1, path_2)
            break 

        elif (current_2 in cost_so_far_1):
            path_1 = reconstruct_path(came_from_1, start, current_2) 
            path_2 = reconstruct_path(came_from_2, goal, current_2)
            combined_path = join_paths(path_1, path_2)
            break
        
        
        # A detailed description of the steps can be found in the implementation of 
        # a_star_search function 
        # Process the nodes from the start side 
        for next in graph.neighbors(current_1): 
            new_cost = cost_so_far_1[current_1] + graph.cost(current_1, next)

            if next not in cost_so_far_1 or new_cost < cost_so_far_1[next]:
                cost_so_far_1[next] = new_cost
                priority = new_cost + manhattan_heuristic(goal, next)
                frontier_1.put(next, priority)
                came_from_1[next] = current_1

        # Process the nodes from the goal side 
        for next in graph.neighbors(current_2): 
            new_cost = cost_so_far_2[current_2] + graph.cost(current_2, next)

            if next not in cost_so_far_2 or new_cost < cost_so_far_2[next]:
                cost_so_far_2[next] = new_cost
                priority = new_cost + manhattan_heuristic(start, next)
                frontier_2.put(next, priority)
                came_from_2[next] = current_2


    return combined_path


# Weighted A-star 
'''
    Weighting sacrifices solution optimality to speed up the search. The larger the weight, 
    the more greedy the search.
    Weighted A* does not provide the optimal path. Rather speeds up the process. z

    The movement cost used for this is: 
    f = g + w2 * h 
    Here, 
    g = g-cost -> Distance from start 
    h = h-cost -> Distance from goal 
    w2 = (1 + e) where e > 0 = How much we consider the heuristic;
    For this algorithm 
    e = 4. So w2 = 5 => Theoretically return us a solution 5 times faster   
'''

def weighted_a_star(graph, start, goal, weight=5): 
    
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
                priority = new_cost + manhattan_heuristic(goal, next) * weight
                frontier.put(next, priority)
                came_from[next] = current

    
    # Return the cost and source dictionary 
    #return came_from, cost_so_far
    return reconstruct_path(came_from, start, goal)


# Dynamically weighted A* 
def dynamic_weighted_astar(graph, start, goal, node_threshold=12, epsilon=2):
    # The weight which  we will prioritise the goal 
    weight = 1 
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

    # Counter to track the depth 
    count = 0
    depth = 1 

    # While the Priority queue is not empty 
    while not frontier.isEmpty():
        # Get the top of queue 
        current = frontier.get()
        count += 1

        # Adjust depth of the search
        # If we have searched all nodes in that level increase depth  
        if count == 4 ** depth:
            depth += 1

        # check if we have reached destination 
        if current == goal: 
            break 
        
        # Dynamically calculate the weight  
        if(depth <= node_threshold): 
            # Dynamic weighting 
            weight = 1 - (depth / node_threshold)
        else:
            weight = 0

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
                priority = new_cost + manhattan_heuristic(goal, next) * (1 + weight * epsilon)
                frontier.put(next, priority)
                came_from[next] = current

    
    # Return the cost and source dictionary 
    #return came_from, cost_so_far
    return reconstruct_path(came_from, start, goal)