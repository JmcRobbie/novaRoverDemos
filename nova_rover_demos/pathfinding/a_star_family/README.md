# A-Star Pathfinding Algoirthm & Variations 
This repository is part of my work for the Melbourne Space Program. In this repository I have implemented variations of the a-start pathfinding algorithm. The purpose is to compare efficiency metrics and have a code base variations.

# Compared Algorithms 
    1. Base A-Star 
    2. Simple Bidirectional A-Star 
    3. Weighted A*
    4. Dynamic Weighted A*  


### Bidirectional A-Star 
Instead of searching from the start to the finish, we can start two searches in parallel―one from start to finish, and one from finish to start. When they meet, or the any of the searches reach their target we get a path. 

Uni-directional A*          |      Bidirectional A*
--------------------|---------------------
![Regular A star](img/unidirectional-a-star.png)   | ![Bidirectional A*](img/bidirectional-astar.png)


Time complexity:
- Regular A*: O(b<sup>d</sup>)
- Bidirectional A*: O(b<sup>d/2</sup>)

Here **b** is the branching factor and **d** is distance of goal vertex from source 

Research Paper: https://arxiv.org/pdf/1703.03868.pdf


### Weighted A* 
With Weighted A* we are making a tradeoff between optimal path and speed. We add a bias towards finding the goal hence the target finding procedure is speed up. However, Weighted A* does not provide the most optimal map. 

Instead of calculating 
    Regular A* => cost = g(n) + h(n)
    Weighted A* =? cost = g(n) + w * h(n)
Formal of defining this.  
    Heuristic calculation, h(w) = h(n) = ε h(n) where ε > 1

Regular A*          |        Weighted A*
--------------------|---------------------
![Regular A star](img/base-astar.gif)   | ![Weighted A*](img/weighted-astar.gif)


As we can see the weighted A* algorithm explores a lower number of nodes and finds our goal quicker. 


Further info: 
- [Stack Overflow](https://stackoverflow.com/questions/44274729/a-search-advantages-of-dynamic-weighting)
- Wikipedia -- [Bounded Relaxation for the A* Search Algorithm](https://en.wikipedia.org/wiki/A*_search_algorithm#/media/File:Astar_progress_animation.gif)




### Dynamic Weighted A* 

Instead of having a static weight for the cost calculation we dynamically calculate the cost function. The feature of this variant is at the start we have a high bias towards finding the goal but as we get deeper into our search we gradually revert to our regular A* algorithm. 

![Weighted A*](img/weighted-astar.gif)

The cost function calculation: 
f(n) = g(n) + (1 + ε * w(n)) * h(n) where, 

![Dynamic weighting equation](img/dynamic-weighting.png)

Here, 

**d(n)** = Depth of search. For my implementation I used a 2D grid hence level of nodes is the depth. For example the neighbors of starting node has a d(n) = 1 and the neighbors of neighbor of starting node has a d(n) = 2

**N** = The estimated distance of the path. For my implementation the highest path can be the range of the sensor for the rover. I assumed a dummy value of 100 nodes as the range of my robot. 


**Note:** I have not extensively tested Dynamic weighted approach and have not found any proper code implementation which I can compare it with. Hence, please take the performance of this with caution. 


Research Paper: https://www.cs.auckland.ac.nz/courses/compsci709s2c/resources/Mike.d/Pohl1973WeightedAStar.pdf



# Implemented Heuristics 

Name of Heuristic | Equation of Heuristic 
------------------|----------------------
Euclidean Distance | h(n) = sqrt( (x<sub>goal</sub> - x)<sup>2</sup> + (y<sub>goal</sub> - y)<sup>2</sup> )
Manhattan Distance | h(n) = abs((x<sub>goal</sub> - x)) + abs((y<sub>goal</sub> - y))
Diagonal Distance | h(n) = max(abs((x<sub>goal</sub> - x)), abs((y<sub>goal</sub> - y)))

 
# Metrics 

#### Map: OpenGrid - Real space without dimensions

Algorithm | # times run | Median (ms) | Mean (ms) | Std. Deviation |   Avg. Path Length 
----------|-------------|-------------|-----------|----------------|---------------------------
Base A*   | 3025        | 23.943662643432617 | 26.743000282728968 | 7.554963703072549 | 62.0
Bidirectional A* | 2960 | 17.917275428771973| 19.237902518865223 | 5.354171559235531 | 62.0
Weighted A*| 2910 | 0.9999275207519531 | 1.4432568730357587 | 0.6828766958337971 | 64.0
Dynamic Weighted A* | 3105 | 8.986949920654297 | 10.285768339983315 | 2.789930903729083 | 64.0

Note: 
- These metrics are meant to give us relative understanding rather than absolute performance 
- The heuristic used for all algorithms is Manhattan Distance 


#### Map: Randomly Weighted Grids of Size 40 by 40

Algorithm | # times run | Median (ms) | Mean (ms) | Std. Deviation |   Avg. Path Length 
----------|-------------|-------------|-----------|----------------|---------------------------
Base A*   | 3016        | 22.937893867492676 | 24.728299056819644 | 5.560186372625823 | 62.0
Bidirectional A* | 3001 | 15.959024429321289| 17.706753054844146 | 4.262223894687466 | 62.0
Weighted A*| 3027 | 0.9982585906982422 | 1.320655486904279 | 0.5098137747095521 | 64.0
Dynamic Weighted A* | 2956 | 3.988981246948242 | 3.995342248186207 | 1.0668236012002967 | 64.0

### Key Insights 
1. Weighted A* is generally much faster than other algorithms but its high bias is a risk and can fail in certain situations. 
2. Dynamic Weighted A* although relatively slower than static weighted in the long run the path length can be more optimal. 
3. Both statically and dynamic weighted A* is faster but the path is not optimal. 
4. Bidirectional generally produces an optimal path but the time consumption is high. 
5. Base A-Star is slower but the path generated is the most optimal 

To run benchmarking yourself, run the following commands 

To see the output of base-a-star and different variants together, simply run. It makes 1-2 mins to run the script due to the high volume of tests.  

```python3 
python benchmark.py
```

# Running instructions 

To see the output of base-a-star and different variants together, simply run 

```python3 
python run.py
```

An additional file has been added if you want to individually alter, test or run any of algorithms

```python3
python tester.py
```

Here are some sample outputs of different algorithms run on the same diagram: 

![Regular A*](img/regular-output.png)

![Bidirectional A*](img/bidirectional-output.png)

![Weighted A*](img/weighted-output.png)


## Different Diagrams 
There are 5 different diagram choices available named diagram1 .... digram5 for testing purposes.You can add maps of your own according to the specifications. To know more about the maps used check the README file inside the 'maze' folder. 

You can select the diagram of your choice 

```python
grid = diagram[You choice of diagram no.]
```

You should also make sure the start and goal coordinates are valid as different dimensions. The diagrams can be found in '/maze/diagrams.py'

```python
start, goal = (x, y), (u, v)
```
The diagrams also vary on the weight of each cell as well as wall configuration. 


**Note**: Any kind contribution is welcome. Please feel free to open a pull request and I will review it as soon as possible.  


# Attributions 

- GIFs used for Weighted and Dynamic A* are taken from Wikipedia 
- Initial SqaureGrid Maze code is taken from [Introduction to A* by Amit](http://theory.stanford.edu/~amitp/GameProgramming/AStarComparison.html#:~:text=A*%20is%20the%20most%20popular,a%20heuristic%20to%20guide%20itself.)