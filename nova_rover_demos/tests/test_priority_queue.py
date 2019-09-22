import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
try:
    from utils.priority_queue import PriorityQueue
except:
    raise
    
import numpy as np


def test_priority_queue(arr):
    output = sorted(arr)
    pq = PriorityQueue()
    
    for i in arr:
        pq.put(i, i)
        
    check = []
    for j in pq.len():
        check.append(pq.get())
    
    print(check, output)
    assert(check == output)


    
arr = np.random.shuffle(range(50))
