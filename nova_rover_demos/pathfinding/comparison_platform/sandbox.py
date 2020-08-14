
    # analytics = [{'runtime': [], 'path_length': []} for algorithm in algorithm_list]

    # maze_x_dimension = 50
    # maze_y_dimension = 50

    # occupancy_grid, start_coordinates, end_coordinates = random_maze(maze_x_dimension, maze_y_dimension, density)

    # for i in range(iterations):
    #     print(i)
    #     '''
    #     Randomly order the operation of the algorithms in the list
    #     '''
    #     pair = list(zip(algorithm_list, analytics))
    #     random.shuffle(pair)
    #     algorithm_list, analytics = zip(*pair)

    #     for algorithm_index in range(len(algorithm_list)):
    #         '''
    #         Compute the path with given algorithm and collect analytic data
    #         '''
    #         t_start = time.time()
    #         path, maze_solved = algorithm_list[algorithm_index](occupancy_grid, start_coordinates, end_coordinates)
    #         t_end = time.time()
    #         '''
    #         Write analytic data to dictionary
    #         '''
    #         analytics[algorithm_index]['runtime'].append(t_end - t_start)

    #         if maze_solved == True:
    #             analytics[algorithm_index]['path_length'].append(len(path))
    #         else:
    #             analytics[algorithm_index]['path_length'] = np.inf

    # '''
    # Average all the quantities in the data analytics over the number of iterations
    # '''
    # for index in range(len(algorithm_list)):
    #     analytics[index]['path_length'] = int( np.average(analytics[index]['path_length']) )

    #     analytics[index]['runtime_variance'] = np.var(analytics[index]['runtime'])
    #     analytics[index]['runtime'] = np.average(analytics[index]['runtime'])

    # algorithm_list = [function.__name__ for function in algorithm_list]
    # return list(zip(algorithm_list, analytics))
