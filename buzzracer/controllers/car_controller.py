# parent class, used as documentation for common function and properties
from common import *
import numpy as np
from buzzracer.extensions.simulators.kinematic_bicycle_cartesian_simulator import KinematicBicycleCartesianSimulator
from buzzracer.extensions.simulators.dynamic_bicycle_cartesian_simulator import DynamicBicycleCartesianSimulator


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
        KinematicBicycleCartesianSimulator.dt = self.car.main.dt
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

    # return control signals
    # throttle, steering
    def control(self):
        throttle = 0.0
        steering = 0.0
        return (throttle, steering)

    # predict car's future trajectory over a short horizon
    # simple baseline method use current control and a kinematic model
    # update predicted_traj vector
    def predict(self):
        # DEBUG plotting
        control = np.array((self.car.steering, self.car.throttle))
        control = np.repeat(np.reshape(control, (1, -1)), self.horizon, 0)
        # kinematic
        expected_trajectory = self.get_kinematic_trajectory(
            self.car.state, control)
        # self.plot_trajectory(expected_trajectory)
        self.predicted_traj = expected_trajectory
        return self.predicted_traj

    def plot_trajectory(self, trajectory):
        if (not self.car.main.visualization.update_visualization.is_set()):
            return
        img = self.car.main.visualization.visualization_img
        for coord in trajectory:
            img = self.car.main.track.draw_circle(
                img, coord, 0.02, color=(0, 0, 0))
        self.car.main.visualization.visualization_img = img
        return

    # debugging functions
    def get_kinematic_trajectory(self, x0, control):
        trajectory = []
        state = x0
        for i in range(control.shape[0]):
            state = KinematicBicycleCartesianSimulator.advance_dynamics(
                state, control[i], self.car, self.main.dt)
            trajectory.append(state)
        return np.array(trajectory)

    def get_dynamic_trajectory(self, x0, control):
        trajectory = []
        state = x0
        for i in range(control.shape[0]):
            state = DynamicBicycleCartesianSimulator.advance_dynamics(
                state, control[i], self.car, self.main.dt)
            trajectory.append(state)
        return np.array(trajectory)
