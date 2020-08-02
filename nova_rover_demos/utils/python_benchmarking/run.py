# Importing the benchmarker 
from src.benchmarker import benchmarker

# Import the functions you want to compare for the 'lib' folder 
from lib.module import *
# Please edit these accordingly, this is an example 
from lib.a_star import *
from lib.a_star_variants import *

# Import any additional dependencies required to fun the program 
# Example 
from lib.maze.diagrams import diagram3


'''
    A list containing the name of the functions you want to compare in the format
    functions = [ f1_name, f2_name, f3_name ... ] 
    
    For example I want to compare different versions of A* algorithms and I have listed
    the names of the ones I want to compare below  
'''
# Example 
functions = [ a_star_search, weighted_a_star, bidirectional_a_star ] # Edit this line 

'''
    List the arguments we want to pass into each function
    We have to list the arguments inside a list and in the order we listed the functions 

    args = [args for f1], [args for f2], [arguments for f3] ...

    For example if wanted to pass a graph, start and goal to and I will pass that as an 
    argument of each A* algorithms. I would do the following way: 
''' 

# Example 
start, goal = (1, 4), (38, 28)
graph = diagram3
# NOTE - the ORDER of the functions above and arguments you put here MUST MATCH!  
args = [graph, start, goal], [graph, start, goal], [graph, start, goal] # Edit this line 

# Passing our functions and arguments into the benchmarker 
benchmarker(functions, args)
