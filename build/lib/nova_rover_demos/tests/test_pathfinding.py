from pathfinding.a_star import a_star_search
from pathfinding.dijkstra import dijkstra_search
from pathfinding.greedy_search import greedy_search

def basic_grid():
    array = [0.0, 0.1, 0.6, 0.8]
    grid = np.random.choice(array, (8, 8))
    return grid

def norm_grid(size=(15, 15)):
    grid = []
    n_rows, n_cols = size
    for i in range(n_rows):
        grid.append([])

        for j in range(n_cols):
            val = np.random.normal(loc=0.2, scale = 0.7)
            while val < 0 or val > 1:
                val = np.random.normal(loc=0.5, scale=0.7)

            grid[i].append(val)

    return grid

def plot_grid(grid, path):
    plt.imshow(grid, cmap='hot', interpolation='nearest')

    x_pos = []
    y_pos = []

    for x in path:
        x_pos.append(x[1])
        y_pos.append(x[0])
    plt.plot(x_pos,y_pos, "ro",color = "Green")

    plt.show()
    
def test_search(grid, search):
    start =  (1, 1)
    goal = (len(grid) - 1, len(grid[0]) - 1)
    path = search(grid, start, goal)
    print(path)
    plot_grid(grid, path)

test_grid = norm_grid()

print("ASTAR")
test_search(test_grid, a_star_search)
print("Greedy")
test_search(test_grid, greedy_search)
test_search(test_grid, dijkstra_search)
