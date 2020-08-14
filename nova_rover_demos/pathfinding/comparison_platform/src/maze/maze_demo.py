import numpy as np


from random_maze import *

xdim = 200
ydim = 200

density = ['sparse', 'light', 'medium', 'heavy']

subplot_ref = [ (0,0), (0,1), (1,0), (1,1)]
fig, axs = plt.subplots(2, 2)



for i in range(len(density)):
    walls, start, end = random_maze(xdim, ydim, density[i])
    maze = np.zeros([xdim, ydim])

    for cell in walls:
        maze[cell] = 1
    axs[subplot_ref[i]].imshow(maze)
    axs[subplot_ref[i]].set_title(f'{density[i]}')

# walls, start, end = random_maze(xdim, ydim, 'super heavy')
# maze = np.zeros([xdim, ydim])

# for cell in walls:
#     maze[cell] = 1
# plt.imshow(maze)
plt.show()
