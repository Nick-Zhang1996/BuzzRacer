''' Replay state history '''
import os
import pickle

import numpy as np

from buzzracer.common import BASEDIR
from buzzracer.cars.car import CarParam
from buzzracer.types import CurvilinearState, CartesianState, Control
from buzzracer.extensions.simulator import Simulator, SimulatorConfig
from buzzracer.extensions.extension import Extension, ExtensionState
from buzzracer.extensions.visualization import Visualization
from buzzracer.sysid.vehicle_dynamics import VehicleDynamics
from buzzracer.controllers.empty_controller import EmptyController

from buzzracer.sysid.kinematic_bicycle_model import KinematicBicycleModelCartesian


class ReplayConfig(SimulatorConfig):
    def __init__(self, main_config):
        super().__init__(main_config)
        self.log_name = None
        ''' Log name from within folder log/ , typically full_state_xxx.p'''
        self.curvilinear = False
        ''' State logged is curvilinear '''
        self.draw_future_traj = False
        self.draw_predicted_traj = False
        self.horizon = 100
        """ Steps to plot into future"""
        self.skip_steps = 0
        ''' Time steps to skip from the beginning '''
        self.prediction_model: VehicleDynamics = 'KinematicBicycleModelCartesian'
        ''' The candidate prediction model, subclass of VehicleDynamics'''


@Extension.register('simulator', ReplayConfig, ExtensionState)
class Replay(Simulator):
    ''' Replay state log, also supports visualizing prediction from an alternative
    dynamics model to visually check difference. '''
    prediction_model = KinematicBicycleModelCartesian

    def __init__(self, config, state):
        super().__init__(config, state)
        self.car_count = 0
        ''' Total car count in this log'''
        self.timestep = 0
        ''' Current time step '''
        self.track = self.main.track
        self.data = None
        ''' Loaded full state log, List of (x,y,heading ...)'''
        Replay.prediction_model = eval(self.config.prediction_model)
        self.print_info(f'prediction model: {Replay.prediction_model}')

    def init(self):
        super().init()
        self.load_log(self.config.log_name)
        self.main.new_state_update.set()
        for car in self.main.cars:
            if not isinstance(car.controller, EmptyController):
                raise RuntimeError("During replay, controller must be EmptyController, edit .xml")
        # for ext in Extension.extensions:
        #     if isinstance(ext, Visualization):
        #         raise RuntimeError("Replay module works with VisualizationGL only")

    def load_log(self, log_name):
        ''' Load full state log '''
        # state dim: [time_steps,  car_count,  (dim_states + dim_action)]
        full_path = os.path.join(BASEDIR, 'outputs', 'logs', log_name)
        self.print_ok(f'opening log file at {full_path}')
        with open(full_path, 'rb') as f:
            self.data = np.array(pickle.load(f)[self.config.skip_steps:])
        self.car_count = self.data.shape[1]
        if len(self.main.cars) != self.car_count:
            self.print_error(
                'The number of cars in log does not match number of cars in config,'
                f' please update config to include {self.car_count} cars')

    def update(self):
        track = self.track
        if self.timestep >= self.data.shape[0]:
            self.main.exit_request.set()
            return

        if self.config.curvilinear:
            for (i, car) in enumerate(self.main.cars):
                curv_state = CurvilinearState(*self.data[self.timestep, i])
                car.state = track.curv_to_cart(curv_state)
        else:
            for (i, car) in enumerate(self.main.cars):
                car.state = CartesianState(*self.data[self.timestep, i, 1:7])
        car.throttle = self.data[self.timestep, i, 8]
        car.steering = self.data[self.timestep, i, 7]

        if self.config.draw_future_traj:
            self.draw_future_trajectory()
        if self.config.draw_predicted_traj:
            self.draw_predicted_trajectory()
        self.main.new_state_update.set()
        self.main.simulator.sim_t += self.main.config.dt
        self.match_real_time()
        self.timestep += 1

    @staticmethod
    def advance_dynamics(state: CurvilinearState | CartesianState,
                         control: Control,
                         car_param: CarParam,
                         dt: float,
                         curvature: float = None) -> CurvilinearState | CartesianState:
        """advance dynamics by dt.

        Args:
            state: state of the car, may be CartesianState or CurvilinearState
            control: (steering,throttle) steering in rad, left positive, throttle in [-1,1],
                    positive indicates acceleration
            car_param: CarParam object, contains information about the car's kinematics,
                also contains car.sim_state for simulators that do not use car.state for update
            dt: Time step to advance dynamics by, unit:seconds
            curvature: signed curvature
        Return:
            state at next time step.
        """
        return Replay.prediction_model.advance_dynamics(state, control, car_param, dt, curvature)

    def draw_future_trajectory(self):
        horizon = self.config.horizon
        color = (0, 0, 0, 255)
        if self.config.curvilinear:
            for (i, _) in enumerate(self.main.cars):
                curvi_states = self.data[self.timestep:self.timestep + horizon, i, 1:8]
                cart_states = []
                for _ in curvi_states:
                    cart_states.append(self.track.curv_to_track(self.data[self.timestep, i]))
                self.main.visualization.draw_polyline(cart_states, color)
        else:
            for (i, _) in enumerate(self.main.cars):
                state_vec = self.data[self.timestep:self.timestep + horizon, i, 1:3]
                self.main.visualization.draw_polyline(state_vec, color)

    def draw_predicted_trajectory(self):
        color = (0, 255, 0, 255)
        dt = self.main.config.dt

        if self.config.curvilinear:
            raise NotImplementedError
        for (i, car) in enumerate(self.main.cars):
            init_state = self.data[self.timestep, i, 1:-2]
            predicted_traj = [init_state]
            for k in range(self.config.horizon):
                states = CartesianState(*predicted_traj[-1])
                control = Control(*self.data[self.timestep + k, i, -2:])
                new_state = self.advance_dynamics(states, control, car.param, dt)
                predicted_traj.append(new_state.to_tuple())

            predicted_traj = np.array(predicted_traj)[:, :2]
            self.main.visualization.draw_polyline(predicted_traj, color)
