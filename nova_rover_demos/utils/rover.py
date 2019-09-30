class Rover:
    def __init__(self,x,y,th):
        # Initial conditions selected arbitrarily
        self.x = x
        self.y = y
        self.theta = th
        self.v = 0
        self.world = []

    def update_state(self,state):
        self.x = state[0]
        self.y = state[1]
        self.theta = state[2]
        self.v = state[3]

    def update_map(self,map):
        self.world = map
    def check_error(self,rover2):
        '''
        Returns the error between the self object an another rover, this is used for checking exit 
        conditions on pose control loops.
        '''
        dx = abs(self.x - rover2.x)
        dy = abs(self.y - rover2.y)
        dth = abs(self.theta - rover2.theta)
        
        return dx*dx+dy*dy+dth*dth