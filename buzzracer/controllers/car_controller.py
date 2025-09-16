''' Base class for all car controllers '''
from __future__ import annotations
import numpy as np

from buzzracer.common import ConfigObject, LogObject
from buzzracer.types import CartesianState, Control
from buzzracer.sysid.kinematic_bicycle_model import KinematicBicycleModelCartesian
from buzzracer.sysid.dynamic_bicycle_model import DynamicBicycleModelCartesian


class CarController(ConfigObject, LogObject):
    def __init__(self, car, config):
        LogObject.__init__(self)
        self.config = config
        self.car = car
        self.main = car.main
        self.track = car.main.track
        # default value
        self.horizon = 30

        self.predicted_traj = []
        super().__init__(config)

    def pre_init(self):
        return

    def post_init(self):
        return

    def init(self):
        # self-reported prediction of future trajectory
        # to be used by opponents for collision avoidance
        self.predict()
        return

    def final(self):
        """called at end of program, override to show statistics."""
        return

    # TODO refactor to use Control
    def control(self) -> tuple[float, float]:
        ''' Main control logic, set output throttle and steering
        Returns:
            control: tuple(throttle, steering)'''
        throttle = 0.0
        steering = 0.0
        return (throttle, steering)

    def predict(self):
        ''' predict car's future trajectory over a short horizon
        simple baseline method use current control and a kinematic model
        update predicted_traj vector
        '''
        control = np.array((self.car.steering, self.car.throttle))
        control_vec = np.repeat(np.reshape(control, (1, -1)), self.horizon, 0)
        # kinematic
        expected_trajectory = self.get_kinematic_trajectory(
            self.car.state, control_vec)
        # self.plot_trajectory(expected_trajectory)
        self.predicted_traj = expected_trajectory
        return self.predicted_traj

    def plot_trajectory(self, trajectory, color=(0, 0, 0)):
        if (not self.car.main.visualization.update_visualization.is_set()):
            return
        img = self.car.main.visualization.visualization_img
        for coord in trajectory:
            img = self.car.main.track.draw_circle(
                img, coord, 0.02, color=color)
        self.car.main.visualization.visualization_img = img
        return

    # debugging functions
    def get_kinematic_trajectory(self, x0, control_vec):
        trajectory = []
        state = CartesianState(*x0)
        for control in control_vec:
            state = KinematicBicycleModelCartesian.advance_dynamics(
                state, Control(*control), self.car, self.main.dt)
            trajectory.append(state)
        return np.array(trajectory)

    def get_dynamic_trajectory(self, x0, control):
        trajectory = []
        state = CartesianState(*x0)
        for i in range(control.shape[0]):
            state = DynamicBicycleModelCartesian.advance_dynamics(
                state, control[i], self.car, self.main.dt)
            trajectory.append(state)
        return np.array(trajectory)
