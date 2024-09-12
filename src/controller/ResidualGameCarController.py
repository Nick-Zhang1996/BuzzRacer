from common import *
from math import isnan,pi,degrees,radians,sin,cos
from controller.CarController import CarController
from controller.PidController import PidController
from third_party.solve_lq_game import solve_lq_game
from extension.simulator.CurvilinearSimulator import CurvilinearSimulator
from util.SymbolicDynamics import SymbolicDynamics
from scipy.linalg import block_diag
import sympy
from util.TimeUtil import TimeUtil
import cv2

from controller.RD3G.CarRacing import CarRacing

class ResidualGameCarController(CarController):
    def __init__(self, car,config):
        super().__init__(car,config)
        self.t = TimeUtil(True)
        self.lqt = TimeUtil(True)
        self.m = 2
        self.n = 4
        self.iterations = 3
        self.draw_prediction = True
        # aggressiveness
        self.alpha = 0

        # for ego agent i -> car 0
        # horizon*m*1
        self.u_i_ref = np.zeros((self.horizon, self.m,1))
        # horizon*n*1
        self.x_i_ref = np.zeros((self.horizon, self.n,1))

        # for ego agent j -> car 1
        self.u_j_ref = np.zeros((self.horizon, self.m,1))
        self.x_j_ref = np.zeros((self.horizon, self.n,1))

        self.horizon = 20
        self.dt = self.main.dt * 2

        self.residual_game = CarRacing()
        self.residual_game.main = self.main
        self.residual_game.track = self.track = self.main.track
        self.residual_game.setup()

        # if true, this controller will control opponent
        self.control_opponent = False

        # NOTE we also need self.track.curvature_fun
        ConfigObject.__init__(self,config)

    def preInit(self):
        if (self.control_opponent):
            self.print_ok('Controller will control opponent')

    def init(self):
        assert(len(self.main.cars)==2)
        self.ego_car = self.car
        for car in self.main.cars:
            if car != self.ego_car:
                self.oppo_car = car
                break

    def solveGame(self, x0, x1, u_ref = None):
        #return: xi_ref, xj_ref, both list of curvilinear states
        g = self.residual_game
        if (u_ref is None):
            g.guess = np.zeros((g.T, g.N, g.m))
        else:
            g.guess = u_ref
        # x0 dim: N*n
        g.x0 = np.vstack([x0.flatten(), x1.flatten()])
        assert(g.x0.shape == (g.N,g.n))
        g.init() # reset parameters, which change between iterations
        u_ref, full_x_ref, has_converged = g.solve(save_gif=False, visualize=False, animate=False)
        if (not has_converged):
            self.print_warning(f'no convergence')
        x0 = full_x_ref[:,0,:]
        x1 = full_x_ref[:,1,:]
        return x0,x1, u_ref, has_converged

    def final(self):
        delta_x = self.ego_car.sim_states - self.oppo_car.sim_states
        self.end_lead_i_j = delta_x[0]
        self.print_info(f'ego car : {self.ego_car.sim_states}')
        self.print_info(f'opponent car : {self.oppo_car.sim_states}')
        self.t.summary()
        self.lqt.summary()

    def isInCollision(self):
        delta_x = self.ego_car.sim_states - self.oppo_car.sim_states
        is_in_collision = np.abs(delta_x[0])<self.opponent_min_distance_s and np.abs(delta_x[2])<self.opponent_min_distance_n
        return is_in_collision

    def control(self):
        self.debug_dict = {}
        # s,v,n,phi
        ctrl0, ctrl1 = self.residualGameControl(self.ego_car.sim_states, self.oppo_car.sim_states)

        # car i
        car_i = self.ego_car
        car_j = self.oppo_car

        #enforce control constraint
        bounded_ctrl,constrained = self.boundControl(ctrl0,car_i)
        car_i.steering = bounded_ctrl[0]
        car_i.throttle = bounded_ctrl[1]

        # car j
        if (self.control_opponent):
            bounded_ctrl,constrained = self.boundControl(ctrl1,car_j)
            car_j.steering = bounded_ctrl[0]
            car_j.throttle = bounded_ctrl[1]

        if (self.draw_prediction):
            self.drawPredictedTrajectory()
        #self.drawDebug()
        return


    def drawDebug(self):
        if (self.main.visualization.update_visualization.is_set()):
            img = self.main.visualization.visualization_img
            car0 = self.ego_car
            car0_coord = car0.states[0:2]
            car0_heading = car0.states[2]
            left, right = self.main.track.preciseTrackBoundary(car0_coord, car0_heading)
            left_pt = [car0_coord[0] + np.cos(car0_heading+np.pi/2)*left, car0_coord[1] + np.sin(car0_heading+np.pi/2)*left]
            right_pt = [car0_coord[0] - np.cos(car0_heading+np.pi/2)*right, car0_coord[1] - np.sin(car0_heading+np.pi/2)*right]
            img = self.main.track.drawPolyline([left_pt,right_pt],img=img)

            self.main.visualization.visualization_img = img

    def drawPredictedTrajectory(self, lineColor=(0,100,100)):
        self.drawTrajectory(self.x_i_ref, lineColor=(0,100,100))
        if (self.control_opponent):
            self.drawTrajectory(self.x_j_ref, lineColor=(0,100,100))
        return

    def drawTrajectory(self, traj=None, lineColor=(0,100,100)):
        #lineColor = (0x22,0x6C,0xFF)
        if (self.main.visualization.update_visualization.is_set()):
            img = self.main.visualization.visualization_img
            predicted_traj = []
            for t in range(self.horizon):
                curvi_states = traj[t]
                cart_states = CurvilinearSimulator.cart2CurvTrack(curvi_states, self.main.track)
                predicted_traj.append(cart_states)

            predicted_traj = np.array(predicted_traj)
            predicted_traj = np.hstack([np.zeros((predicted_traj.shape[0],1)), predicted_traj])
            img = self.main.track.drawTrajectory(np.array(predicted_traj),img,lineColor)
            self.main.visualization.visualization_img = img

