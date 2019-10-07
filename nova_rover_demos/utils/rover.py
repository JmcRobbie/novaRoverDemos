import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
try:
    from pathfinding.heuristic import euclidean_cost
except:
    raise

class Pose:
    # [x, y, yaw]
    def __init__(self, x, y, theta):
        self.x = x
        self.y = y
        self.theta = theta
        
    def get_pose(self):
        return self.x, self.y, self.theta

class Rover(Pose):
    def __init__(self, x, y, th):
        # Initial conditions selected arbitrarily
        super().__init__(x, y, th)
        self.v = 0
        self.world = []

    def update_state(self, state):
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
        
        return dx*dx + dy*dy + dth*dth
    
    def load_waypoints(self, waypoints):
        if waypoints:
            self._waypoints = list(waypoints)
            self._target = self._waypoints.pop(0)
        else:
            print("Error: No waypoints loaded")
    
    def update_waypoints(self):
        if self._waypoints:
            self._target = self._waypoints.pop(0)
        else:
            self._target = None
            
    def target(self):
        return self._target
    
    def set_target_range(self, t_range):
        self.range = t_range
    
    def goal_reached(self):
        if self._dist_to_target() < self.range:
            return True
        return False
    
    def _dist_to_target(self):
        return euclidean_cost((self.x, self.y), (self._target.x, self._target.y)) 
    
    def path_complete(self):
        return self._target == None
    
    def pose(self):
        return Pose(self.x, self.y, self.theta)
    