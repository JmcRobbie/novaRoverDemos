# Path Finding Algorithm Comparison Platform

## Introduction 
This is software platform aimed at comparing the relative performance of various path planning/route finding algorithms. This software tool benchmarks the processing time, peak memory usage and distance of the path found and presents them in a visual manner. We have designed our path finding software in such a way so easily a range of different algorithms can be added to the platform and their performance can be benchmarked.

Currently in the repository there are three classes of pathfinding algorithms are present as default:
1. A* and variants (Heuristic based family) 
2. Pledge (Wall follower family)
3. RRT and variants (Probabilistic family) 


## Usage 

Our goal is to create a scalable easy to use function. To use the comparison platform you simply need to: 

1. Import your algorithm(s) into the comparison platform 
2. Select it inside the run.py file 

### 1. Importing your Algorithm(s)

**Step 1:** You can import the file or folder of the algorithm inside the lib folder. 

[IMAGE]

**Step 2:** Ensure you have fixed any relative imports inside your files. This step is only applicable if you have multiple files are they're importing functions/classes from each other. 

Because our a-star variant file imports a class from a-star and both are under the lib folder, we import in the manner shown in the image above. 

**Step 3:** Write a wrapper class (If applicable). Currently the comparison platform is designed in a way that all functions accept 3 arguments in the order: 
1. List of obstacles (A list of (x, y) tuples)
2. A starting coordinate (A (x, y) format tuple)
3. A goal coordinate (A (x, y) format tuple)

If your algorithm accepts other out formats, you can call the main algorithm with all the necessary arguments inside the wrapper function. 

[IMAGE]

We have put together a very short guide on writing the wrapper functions and you can find [here](). It has a template code for you to get started as well. 

### 2. Configure run.py 

**Step 4:** Import algorithm or wrapper function in the run.py file. 

[IMAGE]

[IMAGE]

Note: Remember to import in the 