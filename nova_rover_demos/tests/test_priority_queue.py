import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
try:
    from utils.priority_queue import PriorityQueue
except:
    raise

import random

def test_priority_queue():
    arr = list(range(50))
    output = arr.copy()
    random.shuffle(arr)
    pq = PriorityQueue()
    
    for i in arr:
        pq.put(i, i)
        
    check = []
    for j in range(pq.len()):
        check.append(pq.get())
    
    assert check == output
