# parent class, used as documentation for common function and properties
from common import PrintObject
import numpy as np
from extension.simulator.KinematicSimulator import KinematicSimulator
from extension.simulator.DynamicSimulator import DynamicSimulator
class CarController(PrintObject):
    def __init__(self, car, config):
        self.config = config
        self.car = car
        self.main = car.main
        self.track = car.main.track
        # default value
        self.horizon = 30

        self.predicted_traj = []
        KinematicSimulator.dt = self.car.main.dt

    def init(self):
        # self-reported prediction of future trajectory
        # to be used by opponents for collision avoidance
        self.predict()
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
        control = np.array((self.car.throttle, self.car.steering))
        control = np.repeat(np.reshape(control,(1,-1)),self.horizon,0)
        # kinematic
        expected_trajectory = self.getKinematicTrajectory( self.car.states, control )
        self.plotTrajectory(expected_trajectory)
        self.predicted_traj = expected_trajectory
        return self.predicted_traj

    def plotTrajectory(self,trajectory):
        if (not self.car.main.visualization.update_visualization.is_set()):
            return
        img = self.car.main.visualization.visualization_img
        for coord in trajectory:
            img = self.car.main.track.drawCircle(img,coord, 0.02, color=(0,0,0))
        self.car.main.visualization.visualization_img = img
        return

    # debugging functions
    def getKinematicTrajectory(self, x0, control):
        trajectory = []
        state = x0
        for i in range(control.shape[0]):
            state = KinematicSimulator.advanceDynamics( state, control[i], self.car)
            trajectory.append(state)
        return np.array(trajectory)
    def getDynamicTrajectory(self, x0, control):
        trajectory = []
        state = x0
        for i in range(control.shape[0]):
            state = DynamicSimulator.advanceDynamics( state, control[i], self.car)
            trajectory.append(state)
        return np.array(trajectory)
