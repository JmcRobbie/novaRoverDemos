class Rover(Pose):
    def __init__(self,x,y,th):
        # Initial conditions selected arbitrarily
        super(x, y, th)
        self.v = 0
        self.world = []

    def update_state(self,state):
        self.x = state[0]
        self.y = state[1]
        self.theta = state[2]
        self.v = state[3]

    def update_map(self, map):
        self.world = map
        
    def check_error(self, pose):
        '''
        Returns the error between the self object and another rover, this is used for checking exit 
        conditions on pose control loops.
        '''
        dx = abs(self.x - pose.x)
        dy = abs(self.y - pose.y)
        dth = abs(self.theta - pose.theta)
        
        return dx*dx+dy*dy+dth*dth
    
    
    def load_waypoints(self, waypoints):
        self._waypoints = list(waypoints)
        self._target = self.waypoints.pop(0)
    
    def update_waypoints(self):
        if self._waypoints:
            self._target = self.waypoints.pop(0)
            return self._target
        
    def get_target_waypoint(self):
        return self._target
    
    def set_target_range(self, t_range):
        self.range = t_range
    
    def goal_reached(self, )
        
class Pose:
    def __init__(self, x, y, theta):
        self.x = x
        self.y = y
        self.theta = theta
    