from maze.diagrams import WeightedGrid 

# utility functions for the map 
def from_id_width(id, width):
    return (id % width, id // width)

# Function to draw proper  
def draw_tile(graph, id, style, width):
    # Default tile design 
    r = '.'
    # Draw the appropriate visual based on condition 
    if 'number' in style and id in style['number']: r  = "%d" % style['number'][id]
    if 'point_to' in style and style['point_to'].get(id, None) is not None:
        (x1, y1) = id
        (x2, y2) = style['point_to'][id]
        # Draw appropriate arrow relative to its parent 
        if x2 == x1 + 1: r = ">"
        if x2 == x1 - 1: r = "<"
        if y2 == y1 + 1: r = "v"
        if y2 == y1 - 1: r = "^"
    if 'start' in style and id == style['start']: r = "A"
    if 'goal' in style and id == style['goal']: r = "Z"
    if 'path' in style and id in style['path']: r = "@"
    if id in graph.walls: r = '#' * width

    # return the proper visual for the grid 
    return r 

# draw the map onto the screen 
def draw_grid(graph, width = 2, **style):
    for y in range(graph.height):
        for x in range(graph.width):
            print("%%-%ds" % width % draw_tile(graph, (x, y), style, width), end = "")
        print()





