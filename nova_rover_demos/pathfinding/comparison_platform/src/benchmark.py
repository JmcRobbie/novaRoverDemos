# Importing necessary files
import math
import time
import random
import statistics
import tracemalloc
import sys
import csv
'''
    The core function which handles the benchmarking of different algorithms 

    @param functions - A tuple containing the functions we want to compare 
    @param args      - A tuple containing the arguments we want to pass into each function
'''


def benchmarker(functions, args):

    # Determining the number of iterations to be made
    iterations = 100 if len(sys.argv) < 2 else int(sys.argv[1])
    # Dictionary to hold the runtime of the comparing functions
    times = {f.__name__: [] for f in functions}
    # Dictionary to hold memory
    peak_memory = {f.__name__: 0 for f in functions}
    # A dictionary to keep track of total path lenth
    avg_path = {f.__name__: 0 for f in functions}
    # Loading the arguments to proper functions
    '''
    args = [ [...], [....], [...]  ]
    '''
    argument_dict = {}
    for i in range(len(functions)):
        argument_dict[functions[i].__name__] = args

    # Running each function randomly around 3000 times
    for i in range(iterations):
        for _ in range(len(functions)):
            # Choose a function randomly from the list and load its arguments
            func = random.choice(functions)
            #func_args = argument_dict[func.__name__]
            # Time its execution start tracing memory allocation
            t0 = time.time()
            tracemalloc.start()
            # Run the functions with the arguments
            # func(*func_args)
            path, status = func(*args)
            # Stop memory tracing
            peak = tracemalloc.get_traced_memory()[1]
            tracemalloc.stop()
            # Stop timer
            t1 = time.time()
            times[func.__name__].append((t1-t0)*1000)
            peak_memory[func.__name__] = peak \
                if peak > peak_memory[func.__name__] else peak_memory[func.__name__]

            avg_path[func.__name__] = len(path) + avg_path[func.__name__]

    return times, peak_memory, avg_path


'''
    A function which outputs the benchmark statistics into a CSV file  

    @param time_stats - Time related statistics 
    @param memory_stats - Memory usage 
    @param path_stats - Contains path lengths 
'''


def print_results(time_stats, memory_stats, path_stats):

    with open('results/benchmark.csv', mode='w') as csv_file:
        field_names = ['Algorithm', 'Mean Runtime',
                       'Std Deviation', 'Peak Memory Usage', 'Avg. Path']
        writer = csv.DictWriter(csv_file, fieldnames=field_names)

        writer.writeheader()
        for name, number in time_stats.items():
            writer.writerow({'Algorithm': name, 'Mean Runtime': statistics.mean(number),
                             'Std Deviation': statistics.stdev(number), 'Peak Memory Usage': memory_stats[name],
                             'Avg. Path': (path_stats[name] / len(number))
                             })
