from src.platform import *
import warnings
'''
Place Algorithm Imports Here
'''

############## IMPORT YOUR ALGORITHM(S) HERE ###########################
from lib.modified_pledge import *
from lib.a_star import *
from lib.a_star_variants import *
from lib.rrt_star_2d import *

#########################################################################
'''
Place the function call for each algorithm to be compared into the list named 'algorithm list'.

Functions should be of the form * My_Function(occupancy_grid, start_position, end_position)
    > occupancy_grid: list of tuples (any length) of coordinates containing walls
    > start_position: length 2 tuple containing x and y starting coordinates
    > end_position: length 2 tuple containing x and y coordinates of the goal coordinates

'''

################## List the algorithms you want to compare here ##############################
algorithm_list = [modified_pledge, a_star,
                  bidirectional_a_star, weighted_a_star, rrt_star_2d_search]


##############################################################################################


# Density keyword describes the density of the environment to be traversed.
# Can be 'heavy', 'medium', 'light' or 'sparse'
density = 'light'


######################################################################
'''
Below this point no input is required from the user :)
'''
######################################################################

warnings.simplefilter("ignore")

compare_algorithms(algorithm_list, density)
