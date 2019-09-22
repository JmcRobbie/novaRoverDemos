import heapq

class PriorityQueue:
    """
    Queue data structure that returns the highest priority item in the queue
    """
    def __init__(self):
        self._elements = []

    """
    Method to return boolean value indicating if PriorityQueue is empty
    """
    def empty(self):
        return len(self._elements) == 0

    """
    Add element item to queue 
    """
    def put(self, item, priority):
        heapq.heappush(self._elements, (priority, item))

    """
    Method to return highest priority item in queue 
    @param state : current state in board 
    @return      : highest priority item in queue if start is none, else the item with the same state 
    """
    def get(self, state=None):
        return heapq.heappop(self._elements)[1]

    """
    Return length of queue
    """
    def len(self):
        return len(self._elements)

    """
    Method to return boolean value indicating if item is contained in the queue 
    """
    def __contains__(self, other):
        for item in self._elements:
            if other == item[1]:
                return True
        return False

    """
    Method to return the index of an item if it occurs in the queue 
    """
    def index(self, item):
        for index, elem in enumerate(self._elements):
            if item == elem[1]:
                return index
        return -1
