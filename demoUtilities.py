class rover:
    def __init__(self,x,y,th):
        # Initial conditions selected arbitrarily
        self.x = x
        self.y = y
        self.theta = th
        self.v = 0
        self.world = []
    def updatestate(self,state):
        self.x = state[0]
        self.y = state[1]
        self.theta = state[2]
        self.v = state[3]
    def updateMap(self,map):
        self.world = map
