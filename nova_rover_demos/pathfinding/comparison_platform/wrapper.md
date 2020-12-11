# Wrapper function guidelines 

Wrapper function helps us run the comparison platform in a systematic way. You **do not** need a wrapper function if you algorithm has both the following characteristics: 
- Accepts only list of obstacles, start coordinate & end coordinate as function parameters 
- Returns the path as a list of tuples and a boolean indicating path was found or not 

If not then you can either tweak your existing function or put it inside a wrapper function. Here is a template for the wrapper function: 

```python
def WRAPPER_FUNCTION(obstacle_list, start, end):
    
    # Instantiate appropriate attributes if necessary
    
    
    try:
        # Call the main algorithm
        # It should return the path as a list of coordinates (tuples)
        path = YOUR_ALGORITHM_CALL
        maze_solved = True
    except:
        path = start
        maze_solved = False

    # Return a list of tuples as path and
    # a boolean indicating maze solved or not
    return path, maze_solved

```

You should create this wrapper inside the **lib** folder and afterwards pass its name in `run.py` file. 