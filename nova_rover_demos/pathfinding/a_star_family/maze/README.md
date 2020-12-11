# Maps 

This repository allows implementation of two different format of maps. This README is designed provide information regarding the maps. 

## Real Space - Open Grid 
This grid is designed to be combined with FastSLAM 2 to provide autonomous navigation for the Mars Rover. This map goes for provide any specific grid dimensions. 
All this map needs is list of tuples which are the cartesian coordinates of the obstacles/walls. This is empty by default. 

This type of map is implemented through the following class: 
```python
class OpenGrid:
    def __init__(self):
        self.walls = []
        self.weights = {}
``` 
As we can see the default walls and weights are set to empty. The default value for weights of cell is 1 otherwise specified. 

Important things to consider: 
- Since there isn't a size, not having a path can make the program forever 
- The walls have to assigned as a list of tuples 

## Dimension Specific Grid 
This grid requires us define a dimension of the map at the beginning. Otherwise it's the same as the OpenGrid class.

This class is defined as: 
```python
class WeightedGrid(SquareGrid):
    def __init__(self, width, height):
        super().__init__(width, height)
        self.weights = {}
```

Implementation of SquareGrid: 
```python
class SquareGrid:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.walls = []
```