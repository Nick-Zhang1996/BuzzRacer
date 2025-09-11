''' Replay state history '''
import os
import pickle

import numpy as np
from scipy.interpolate import splev

from buzzracer.types import CurvilinearState, CartesianState
from buzzracer.common import BASEDIR
from buzzracer.utilities.execution_timer import ExecutionTimer
from buzzracer.extensions.simulators.dynamic_simulator import DynamicSimulator
from buzzracer.extensions.simulator import Simulator
from buzzracer.sysid.gp_model import GpModel


class Replay(Simulator):
    ''' Replay state log, also supports visualizing prediction from an alternative
    dynamics model to visually check difference. '''

    def __init__(self):
        super().__init__(handle_name='replay')
        self.t = ExecutionTimer(True)
        self.car_count = 0
        ''' Total car count in this log'''
        self.timestep = 0
        ''' Current time step '''
        self.curvilinear = None
        ''' State logged is curvilinear '''
        self.log_name = None
        ''' Log name from within folder log/ , typically full_state_xxx.p'''
        self.skip = 0
        ''' Time steps to skip from the beginning '''
        self.draw_future_traj = False
        self.draw_predicted_traj = False
        self.track = self.main.track
        # TODO move this to setting
        # self.prediction_model = DynamicBicycleModel()
        # self.prediction_model = KinematicBicycleModel()
        self.prediction_model = GpModel()
        ''' The candidate prediction model, subclass of VehicleDynamics'''

        self.data = None
        ''' Loaded full state log, List of (x,y,heading ...)'''

        DynamicSimulator.dt = self.main.dt

    def init(self):
        super().init()
        self.load_log(self.log_name)
        self.main.new_state_update.set()

    def load_log(self, log_name):
        ''' Load full state log '''
        # state dim: [time_steps,  car_count,  (dim_states + dim_action)]
        full_path = os.path.join(BASEDIR, 'log', log_name)
        self.print_ok(f'opening log file at {full_path}')
        with open(full_path, 'rb') as f:
            self.data = pickle.load(f)[self.skip:]
        self.car_count = self.data.shape[1]
        if len(self.main.cars) != self.car_count:
            self.print_error(
                'The number of cars in log does not match number of cars in config,'
                f' please update config to include {self.car_count} cars')

    def CurvilinearToCartesian(self,
                               state: CurvilinearState) -> CartesianState:
        # TODO verify this is right, dimension
        pos = np.array(
            splev(state.progress % self.track.raceline_len_m,
                  self.track.raceline_s,
                  der=0))
        A = np.array([[0, -1], [1, 0]])
        tangent = np.array(
            splev(state.progress % self.track.raceline_len_m,
                  self.track.raceline_s,
                  der=1))
        track_heading = np.arctan2(tangent[1], tangent[0])
        lateral = A @ (tangent / np.linalg.norm(tangent))
        car_pos = pos + state.lateral_err * lateral
        x, y = car_pos
        heading = state.heading_err + track_heading
        # NOTE we ignored second order curvature in reference curve
        return CartesianState(x=x,
                              y=y,
                              heading=heading,
                              v_forward=state.v_forward,
                              v_sideway=state.v_sideway,
                              omega=state.rel_omega)

    def update(self):
        if self.timestep >= self.data.shape[0]:
            self.main.exit_request.set()
            return

        if self.curvilinear:
            for (i, car) in enumerate(self.main.cars):
                car.state = self.CurvilinearToCartesian(
                    self.data[self.timestep, i])
        else:
            for (i, car) in enumerate(self.main.cars):
                car.state = tuple(self.data[self.timestep, i, 1:7].flatten())
        car.throttle = self.data[self.timestep, i, 8]
        car.steering = self.data[self.timestep, i, 7]

        if self.draw_future_traj:
            self.draw_future_trajectory()
        self.t.s()
        self.t.s('draw_predicted_trajectory')
        if self.draw_predicted_traj:
            self.draw_predicted_trajectory()
        self.t.e('draw_predicted_trajectory')
        self.t.e()
        self.main.new_state_update.set()
        self.main.simulator.sim_t += self.main.dt
        self.match_real_time()
        self.timestep += 1

    def final(self):
        self.t.summary()

    def draw_future_trajectory(self, horizon=0.5):
        lineColor = (255, 0, 0)
        if self.main.visualization.update_visualization.is_set():
            img = self.main.visualization.visualization_img
            if self.curvilinear:
                for (i, _) in enumerate(self.main.cars):
                    curvi_states = self.data[self.timestep:self.timestep +
                                             int(horizon / self.main.dt), i, :]
                    cart_states = []
                    for _ in curvi_states:
                        cart_states.append(
                            self.CurvilinearToCartesian(
                                self.data[self.timestep, i]))
                    img = self.main.track.draw_trajectory(
                        cart_states, img, lineColor)
            else:
                for (i, _) in enumerate(self.main.cars):
                    img = self.main.track.draw_trajectory(
                        self.data[self.timestep:self.timestep +
                                  int(horizon / self.main.dt), i, :], img,
                        lineColor)
            self.main.visualization.visualization_img = img

    def draw_predicted_trajectory(self, horizon=0.5):
        lineColor = (0, 255, 0)
        if self.main.visualization.update_visualization.is_set():
            img = self.main.visualization.visualization_img
            if self.curvilinear:
                raise NotImplementedError
            else:
                for (i, car) in enumerate(self.main.cars):
                    init_state = self.data[self.timestep, i, 1:-2]
                    predicted_traj = [init_state]
                    for _ in range(int(horizon / self.main.dt)):
                        states = predicted_traj[-1]
                        control = self.data[self.timestep +
                                            len(predicted_traj) - 1, i, -2:]
                        new_state = self.prediction_model.advance_dynamics(
                            states, control, car, self.main.dt)
                        predicted_traj.append(new_state)

                    predicted_traj = np.array(predicted_traj)
                    predicted_traj = np.hstack([
                        np.zeros((predicted_traj.shape[0], 1)), predicted_traj
                    ])
                    img = self.main.track.draw_trajectory(
                        np.array(predicted_traj), img, lineColor)
            self.main.visualization.visualization_img = img
