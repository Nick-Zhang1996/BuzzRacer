# parent class, used as documentation for common function and properties
from common import PrintObject
import numpy as np
import math
from extension.simulator.KinematicSimulator import KinematicSimulator
from extension.simulator.DynamicSimulator import DynamicSimulator
from extension.simulator.CalebDynamicSimulator import CalebDynamicSimulator
from extension.simulator.KinetoDynamicSimulator import KinetoDynamicSimulator
from extension.Simulator import SimulatorType

from track.RCPTrack import RCPTrack
from scipy.signal import savgol_filter
from scipy.interpolate import splev


class CarController(PrintObject):
    def __init__(self, car):
        self.car = car
        self.main = car.main
        self.track = car.main.track

        # self-reported prediction of future trajectory
        # to be used by opponents for collision avoidance
        self.prediction_horizon = 30
        # n*2, n being prediction horizon
        self.predicted_traj = []
        KinematicSimulator.dt = self.car.main.dt

        if self.main.SimulatorType == SimulatorType.KinetoDynamicSimulator:
            n_steps = 1000
            uu = np.linspace(0, self.track.track_length_grid, n_steps + 1)
            self.maxDistance = self.track.uToS(uu[-1])
            discretized_raceline_len = 1024

            _norm = lambda x: np.linalg.norm(x, axis=0)
            self.ss = np.linspace(0, self.track.raceline_len_m, discretized_raceline_len)
            dr = np.array(splev(self.ss, self.track.raceline_s, der=1))
            ddr = vec_curvature = np.array(splev(self.ss, self.track.raceline_s, der=2))
            der = np.array(splev(self.ss, self.track.raceline_s, der=1))

            curv = 1.0 / (_norm(dr) ** 3 / (_norm(dr) ** 2 * _norm(ddr) ** 2 - np.sum(dr * ddr, axis=0) ** 2) ** 0.5)

            # curvature needs to be signed to indicate whether signage target angular velocity
            # a cross product gives right signage for omega, this is indep of track direction since it's calculated based off vehicle orientation
            # cross_curvature = der[0, :] * vec_curvature[1, :] - der[1, :] * vec_curvature[0, :]
            cross_curvature = der[0] * vec_curvature[1] - der[1] * vec_curvature[0]

            k = curv
            k_sign = cross_curvature
            k_signed = np.copysign(k, k_sign)
            self.k_signed_smooth = savgol_filter(k_signed, 75, 2)

    def init(self):
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
        control = np.repeat(np.reshape(control,(1,-1)),self.prediction_horizon,0)
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
        if self.main.simulator.advanceDynamics == DynamicSimulator:
            for i in range(control.shape[0]):
                state = self.main.simulator.advanceDynamics(state, control[i], self.car)
                trajectory.append(state)
        elif self.main.simulator.advanceDynamics == KinetoDynamicSimulator:
            for i in range(control.shape[0]):
                curvature = self.getCurvature(state[4])
                state = self.main.simulator.advanceDynamics(state, control[i], curvature, self.car)
                zeta = state[4] % self.maxDistance  # need to wrap zeta for interpolation function
                u = self.track.sToU(zeta)
                xRef, yRef = splev(u, self.track.raceline, der=0)
                der = np.array(splev(u, self.track.raceline, der=1))
                headingRef = math.atan2(der[1], der[0])
                n = state[5]
                x = xRef - n * np.sin(headingRef)
                y = yRef + n * np.cos(headingRef)
                trajectory.append((x, y, 0, 0, 0, 0))
        return np.array(trajectory)

    # guess is a zeta value
    def getCurvature(self, zeta):
        idx = np.searchsorted(self.ss, zeta, side="left")
        if idx > 0 and (idx == len(self.ss) or math.fabs(zeta - self.ss[idx - 1]) < math.fabs(zeta - self.ss[idx])):
            return self.k_signed_smooth[idx - 1]
        else:
            return self.k_signed_smooth[idx]