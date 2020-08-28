import statistics
import matplotlib.pyplot as plt
import numpy as np


# The function responsible for displaying the plots in the screen
def visualiser(time_stats, memory_stats, path_stats):

    # Converting to appropriate data
    func_names = []
    performance = []
    error = []
    peak_memory = []
    avg_path = []

    for name, number in time_stats.items():
        func_names.append(name)
        performance.append(statistics.mean(number))
        error.append(statistics.stdev(number))
        peak_memory.append(memory_stats[name])
        avg_path.append(path_stats[name] / len(number))

    y_pos = np.arange(len(func_names))

    # Plotting the runtime performance
    fig1 = plt.figure(figsize=(10, 10))
    ax1 = fig1.add_subplot(211)

    ax1.barh(y_pos, performance, xerr=error, align='center',
             color='green', ecolor='black')
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(func_names)
    # Read labels top to bottom
    ax1.invert_yaxis()
    # Labels
    ax1.set_xscale('log')
    ax1.set_xlabel('Mean Runtime (ms)')
    ax1.set_title('Runtime Comparison')

    # Plotting path visuals
    ax_path = fig1.add_subplot(212)
    ax_path.barh(y_pos, avg_path, align='center',
                 color='purple', ecolor='black')
    # Setting y-axis labels
    ax_path.set_yticks(y_pos)
    ax_path.set_yticklabels(func_names)
    # Adding x-axis labels
    # Read labels top to bottom
    ax_path.invert_yaxis()
    ax_path.set_xlabel('Path Length')
    ax_path.set_title('Distance Travelled')

    # Adding some padding between layouts
    fig1.tight_layout(pad=4.0)

    # Plotting the memory performance
    fig2 = plt.figure(figsize=(10, 10))
    ax2 = fig2.add_subplot()

    ax2.barh(y_pos, peak_memory, align='center')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(func_names)
    # Read labels top to bottom
    ax2.invert_yaxis()
    # Labels
    ax2.set_xlabel('Peak Memory Use (KB)')
    ax2.set_title('Memory Usage Comparison')

    fig2.tight_layout()

    # Show the plot
    plt.show()


# A function to draw the grid with path found by each of the algorithms
def plot_diagram(functions, args, maze_x, maze_y):

    # Loop through all the algorithms
    for func in functions:
        path, status = func(*args)

        # Creating an identify matrix of given dimensions
        grid = np.ones([maze_x, maze_y])

        # Populate different kinds of grids
        for i in args[0]:
            grid[i] = 0

        for j in path:
            grid[j] = 2

        grid[path[0]] = 3
        grid[path[-1]] = 4

        # Create a figure and save it
        plt.imshow(grid.T)
        plt.colorbar()
        filename = "results/" + func.__name__ + ".pdf"
        plt.savefig(filename)
        plt.close()
