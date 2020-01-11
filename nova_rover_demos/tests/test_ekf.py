import sys
import os
import unittest
from numpy.testing import *

sys.path.append(os.path.dirname(os.path.abspath('')) + "/../")

try:
    from nova_rover_demos.localisation_mapping.ekf import *
except:
    raise


class TestEKF(unittest.TestCase):

    def setUp(self):
        self.u = np.array([[0.5], [1.5]])
        self.Q = np.diag([0.1, 0.1, np.deg2rad(1.0), 1.0]) ** 2
        self.R = np.diag([1.0, 1.0]) ** 2

        #  Simulation parameter
        self.INPUT_NOISE = np.diag([1.0, np.deg2rad(30.0)]) ** 2
        self.GPS_NOISE = np.diag([0.5, 0.5]) ** 2

        self.DT = 0.1  # time tick [s]
        self.SIM_TIME = 50.0  # simulation time [s]

        # State Vector [x y yaw v]'
        self.xTrue = np.zeros((4, 1))
        self.xEst = np.zeros((4, 1))
        self.PEst = np.eye(4)

        self.xDR = np.zeros((4, 1))  # Dead reckoning

    def test_observation(self):
        xTrue, z, xd, ud = observation(self.xTrue, self.xDR, self.u)
        # testing for array equality with functions from numpy.testing
        assert_almost_equal(xTrue, [[0.05], [0.0], [0.15], [0.5]])
        # no point to test the other three outputs because noise is added to produce random values

    #         assert_almost_equal(z, [[ 0.88849999], [-0.21255136]])
    #         assert_almost_equal(xd, [[0.13807903], [0.0], [0.12893121], [1.38079034]])
    #         assert_almost_equal(ud, [[1.38079034], [1.28931208]])

    def test_ekf_estimation(self):
        xTrue, z, xd, ud = observation(self.xTrue, self.xDR, self.u)
        xEst, PEst = ekf_estimation(self.xEst, self.PEst, z, self.u)
        #         assert_almost_equal(xEst, [[ 0.4733085 ], [-0.10671596], [ 0.14166538], [ 0.53947135]])
        assert_almost_equal(PEst, [[5.04909295e-01, 2.72603352e-04, -3.71274863e-03, 4.89490633e-02],
                                   [2.72603352e-04, 5.03146790e-01, 2.45657411e-02, 7.39792735e-03],
                                   [-3.71274863e-03, 2.45657411e-02, 9.99062381e-01, 6.85223807e-20],
                                   [4.89490633e-02, 7.39792735e-03, 2.37090963e-20, 1.99504950e+00]])

# driver code for testing
# unittest.main(argv=['first-arg-is-ignored'], exit=False)
