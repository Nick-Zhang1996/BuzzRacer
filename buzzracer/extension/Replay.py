# replay state history
from util.timeUtil import ExecutionTimer
from extension.simulator.DynamicSimulator import DynamicSimulator
from time import time
import os
from common import *
import pickle
from extension.Extension import Extension
from extension import Simulator
from math import atan2, radians, degrees, sin, cos, pi, tan, copysign, asin, acos, isnan, atan
from scipy.interpolate import splprep, splev, CubicSpline, interp1d
import matplotlib.pyplot as plt
from sysid.tire import tire_curve
from sysid.gaussian_process.gpModel import MultitaskDeepGP
# sketchy
import sys
import torch
sys.path.append('/home/nickzhang/rcvip/src/sysid/gaussian_process')


class Replay(Simulator):
    def __init__(self):
        super().__init__(handle_name='replay')
        self.t = ExecutionTimer(True)
        self.car_count = 0
        self.timestep = 0
        self.curvilinear = None
        self.log_name = None
        self.skip = 0
        self.draw_future_traj = False
        self.draw_predicted_traj = False
        # rcvip
        self.basedir = self.main.basedir
        self.track = self.main.track
        # TODO move this to setting
        # self.prediction_model = DynamicBicycleModel()
        # self.prediction_model = KinematicBicycleModel()
        self.prediction_model = GpModel()

        # DEBUG
        DynamicSimulator.dt = main.dt

    def init(self):
        super().init()
        if (self.curvilinear):
            self.load_curvilinear_log(self.log_name)
        else:
            self.load_cartesian_log(self.log_name)
        self.main.new_state_update.set()
        # DEBUG
        # lateral_err = self.data[:,0,1]
        # plt.plot(lateral_err)
        # plt.show()
        # breakpoint()

    def load_curvilinear_log(self, log_name):
        # time_steps * cars * (states + action)
        full_path = os.path.join(self.basedir, log_name)
        self.print_ok(f'opening file at {full_path}')
        with open(full_path, 'rb') as f:
            self.data = pickle.load(f)
        self.data = self.data[self.skip:]
        # create cars
        self.car_count = self.data.shape[1]
        assert (len(self.main.cars) == self.car_count)

    def load_cartesian_log(self, log_name):
        full_path = os.path.join(self.basedir, log_name)
        self.print_ok(f'opening file at {full_path}')
        with open(full_path, 'rb') as f:
            self.data = np.array(pickle.load(f))
        self.data = self.data[self.skip:]
        # create cars
        # data dimension: timestep, cars, state
        self.car_count = self.data.shape[1]
        if (len(self.main.cars) != self.car_count):
            self.print_error(
                f'number of cars in log does not match number of cars in config, please update config to include {self.car_count} cars')

    def load_rcp_track(self):
        N, X, Y, s, phi, kappa, diff_s, d_upper, d_lower, border_angle_upper, border_angle_lower = self.track.get_orca_style_track()

        self.N = N
        self.X = X
        self.Y = Y
        self.s = s
        self.phi = phi
        self.kappa = kappa
        self.diff_s = diff_s

        self.d_upper = d_upper
        self.d_lower = d_lower
        # not really used
        self.border_angle_upper = border_angle_upper
        self.border_angle_lower = border_angle_lower
        return

    def CurvilinearToCartesian(self, state):
        progress, lateral_err, rel_heading, v_forward, v_sideways, omega, throttle, steering = state
        # TODO verify this is right, dimension
        pos = np.array(splev(progress % self.track.raceline_len_m,
                       self.track.raceline_s, der=0))
        A = np.array([[0, -1], [1, 0]])
        tangent = np.array(
            splev(progress % self.track.raceline_len_m, self.track.raceline_s, der=1))
        track_heading = np.arctan2(tangent[1], tangent[0])
        lateral = A @ (tangent/np.linalg.norm(tangent))
        car_pos = pos + lateral_err * lateral
        x, y = car_pos
        heading = rel_heading + track_heading
        # print(f'tangent = {tangent}')
        # print(f'lateral = {lateral}')
        # print(f'track_heading = {track_heading}')
        return (x, y, heading, v_forward, v_sideways, omega)

    def update(self):
        if (self.timestep >= self.data.shape[0]):
            self.main.exit_request.set()
            return

        if (self.curvilinear):
            for (i, car) in enumerate(self.main.cars):
                car.states = self.CurvilinearToCartesian(
                    self.data[self.timestep, i])
        else:
            for (i, car) in enumerate(self.main.cars):
                car.states = tuple(self.data[self.timestep, i, 1:7].flatten())
        car.throttle = self.data[self.timestep, i, 8]
        car.steering = self.data[self.timestep, i, 7]

        if (self.draw_future_traj):
            self.draw_future_trajectory()
        self.t.s()
        self.t.s('draw_predicted_trajectory')
        if (self.draw_predicted_traj):
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
        if (self.main.visualization.update_visualization.is_set()):
            img = self.main.visualization.visualization_img
            if (self.curvilinear):
                for (i, car) in enumerate(self.main.cars):
                    curvi_states = self.data[self.timestep:self.timestep +
                                             int(horizon/self.main.dt), i, :]
                    cart_states = []
                    for state in curvi_states:
                        cart_states.append(self.CurvilinearToCartesian(
                            self.data[self.timestep, i]))
                    img = self.main.track.draw_trajectory(
                        cart_states, img, lineColor)
            else:
                for (i, car) in enumerate(self.main.cars):
                    img = self.main.track.draw_trajectory(
                        self.data[self.timestep:self.timestep + int(horizon/self.main.dt), i, :], img, lineColor)
            self.main.visualization.visualization_img = img

    def draw_predicted_trajectory(self, horizon=0.5):
        lineColor = (0, 255, 0)
        if (self.main.visualization.update_visualization.is_set()):
            img = self.main.visualization.visualization_img
            if (self.curvilinear):
                print_error('not implemented')
            else:
                for (i, car) in enumerate(self.main.cars):
                    init_state = self.data[self.timestep, i, 1:-2]
                    predicted_traj = [init_state]
                    for t in range(int(horizon/self.main.dt)):
                        states = predicted_traj[-1]
                        control = self.data[self.timestep +
                                            len(predicted_traj)-1, i, -2:]
                        new_state = self.prediction_model.advance_dynamics(
                            states, control, car, self.main.dt)
                        predicted_traj.append(new_state)

                    predicted_traj = np.array(predicted_traj)
                    predicted_traj = np.hstack(
                        [np.zeros((predicted_traj.shape[0], 1)), predicted_traj])
                    img = self.main.track.draw_trajectory(
                        np.array(predicted_traj), img, lineColor)
            self.main.visualization.visualization_img = img


class VehicleDynamics:
    """parent class for vehicle dynamics model."""

    def __init__(self):
        # either cartesian(False) or curvilinear (true)
        # curvilinear:
        # progress, lateral_err, rel_heading, v_forward, v_sideways, omega,throttle,steering = state
        # cartesian:
        # x,y,heading,v_forward,v_sideway,omega = car.states
        self.curvilinear = None

    def advance_dynamics(car_states, control, car, dt):
        '''
        return states at next timestep
        car_states: differs depending on self.cartesian, 
        control:steering,throttle
        '''
        return car_states


class DynamicBicycleModel(VehicleDynamics):
    def __init__(self):
        # either cartesian(False) or curvilinear (true)
        # curvilinear:
        # progress, lateral_err, rel_heading, v_forward, v_sideways, omega,throttle,steering = state
        # cartesian:
        # x,y,heading,v_forward,v_sideway,omega = car.states
        self.curvilinear = False

    def core_dynamics(self, core_states, control, car, dt):
        '''
        input: core_states = (vx,vy,omega) control = (steering,throttle)
        output: (d_vx, d_vy, d_omega)
        TODO can try vy, d_vy, omega, d_omega
        '''
        lf = car.lf
        lr = car.lr
        L = car.L

        Iz = car.Iz
        m = car.m
        vx, vy, omega = core_states
        steering, throttle = control

        # for small longitudinal velocity use kinematic model
        if (vx < 0.05):
            beta = atan(lr/L*tan(steering))
            def norm(a, b): return (a**2+b**2)**0.5
            # motor model
            d_vx = 6.17*(throttle - vx/15.2 - 0.333)
            d_vy = (norm(vx, vy)*sin(beta) - vy)/dt
            d_omega = (vx/L*tan(steering) - omega)/dt

        else:
            slip_f = -np.arctan((omega*lf + vy)/vx) + steering
            slip_r = np.arctan((omega*lr - vy)/vx)

            # Ffy = Df * np.sin( C * np.arctan(B *slip_f)) * 9.8 * lr / (lr + lf) * m
            # Fry = Dr * np.sin( C * np.arctan(B *slip_r)) * 9.8 * lf / (lr + lf) * m
            Ffy = tire_curve(slip_f) * m * 9.8 * lr/(lr+lf)
            Fry = 1.15*tire_curve(slip_r) * m * 9.8 * lf/(lr+lf)

            # Dynamics
            # d_vx = 1.0/m * (Frx - Ffy * np.sin( steering ) + m * vy * omega)
            d_vx = 6.17*(throttle - vx/15.2 - 0.333)
            d_vy = 1.0/m * (Fry + Ffy * np.cos(steering) - m * vx * omega)
            d_omega = 1.0/Iz * (Ffy * lf * np.cos(steering) - Fry * lr)
        return (d_vx, d_vy, d_omega)

    def advance_dynamics(self, car_states, control, car, dt):
        """advance vehicle dynamics NOTE using car frame origined at CG with x
        pointing forward, y leftward this method does NOT update
        car.sim_states, only returns a sim_state this is to make itself useful
        for when update is not necessary."""
        x, y, psi, v_forward, v_sideway, d_psi = car_states
        lf = car.lf
        lr = car.lr
        L = car.L

        Iz = car.Iz
        m = car.m

        # NOTE here vx = vf, vy = vs, different convention
        x, y, heading, vx, vy, omega = car_states
        steering, throttle = control
        d_vx, d_vy, d_omega = self.core_dynamics(
            (vx, vy, omega), control, car, dt)

        # discretization
        vx = vx + d_vx * dt
        vy = vy + d_vy * dt
        omega = omega + d_omega * dt

        # back to global frame
        vxg = vx*cos(heading)-vy*sin(heading)
        vyg = vx*sin(heading)+vy*cos(heading)

        # update x,y, heading
        x += vxg*dt
        y += vyg*dt
        heading += omega*dt + 0.5 * d_omega * dt * dt

        car_states = x, y, heading, vx, vy, omega
        return np.array(car_states)


class KinematicBicycleModel(VehicleDynamics):
    def __init__(self):
        # either cartesian(False) or curvilinear (true)
        # curvilinear:
        # progress, lateral_err, rel_heading, v_forward, v_sideways, omega,throttle,steering = state
        # cartesian:
        # x,y,heading,v_forward,v_sideway,omega = car.states
        self.curvilinear = False

    def advance_dynamics(self, car_states, control, car, dt):
        lr = car.lr
        lf = car.lf

        '''
        throttle = np.clip(throttle, -1.0, 1.0)
        steering = np.clip(throttle, -radians(27), radians(27))
        '''
        x, y, heading, v_forward, v_sideway, omega = car_states
        v = v_forward
        # slow down if car is in collision
        '''
        if (car.in_collision):
            v *= 0.9
        '''
        throttle = control[1]
        steering = control[0]

        beta = np.arctan(np.tan(steering) * lr / (lf+lr))
        dXdt = v * np.cos(heading + beta)
        dYdt = v * np.sin(heading + beta)
        dvdt = 6.17*(throttle - v/15.2 - 0.333)
        omega = dheadingdt = v/lr*np.sin(beta)

        x += dt * dXdt
        y += dt * dYdt
        v += dt * dvdt
        heading += dt * dheadingdt

        v_forward = v
        v_sideway = 0
        car_states = x, y, heading, v_forward, v_sideway, omega
        return np.array(car_states)


class GpModel(VehicleDynamics):
    def __init__(self):
        # either cartesian(False) or curvilinear (true)
        # curvilinear:
        # progress, lateral_err, rel_heading, v_forward, v_sideways, omega,throttle,steering = state
        # cartesian:
        # x,y,heading,v_forward,v_sideway,omega = car.states
        self.curvilinear = False

        model_filename = '/home/nickzhang/rcvip/src/sysid/gaussian_process/model.p'
        input_dim = 5
        output_dim = 3
        self.model = MultitaskDeepGP((100, input_dim), output_dim)
        self.model.load_state_dict(torch.load(model_filename))

    def core_dynamics(self, core_states, control, car, dt):
        '''
        input: core_states = (vx,vy,omega) control = (steering,throttle)
        output: (d_vx, d_vy, omega)
        '''
        lf = car.lf
        lr = car.lr
        L = car.L

        Iz = car.Iz
        m = car.m
        vx, vy, omega = core_states
        steering, throttle = control

        # for small longitudinal velocity use kinematic model
        if (vx < 0.05):
            beta = atan(lr/L*tan(steering))
            def norm(a, b): return (a**2+b**2)**0.5
            # motor model
            d_vx = 6.17*(throttle - vx/15.2 - 0.333)
            d_vy = (norm(vx, vy)*sin(beta) - vy)/dt
            # d_omega =
            omega = vx/L*tan(steering)

        else:
            model_input = torch.Tensor(core_states+tuple(control)).unsqueeze(0)
            mean, var = self.model.predict(model_input)
            output = mean.numpy()
            d_vx = output[0, 0]
            d_vy = output[0, 1]
            omega = output[0, 2]
        return (d_vx, d_vy, omega)

    def advance_dynamics(self, car_states, control, car, dt):
        """advance vehicle dynamics NOTE using car frame origined at CG with x
        pointing forward, y leftward this method does NOT update
        car.sim_states, only returns a sim_state this is to make itself useful
        for when update is not necessary."""
        x, y, psi, v_forward, v_sideway, d_psi = car_states
        lf = car.lf
        lr = car.lr
        L = car.L

        Iz = car.Iz
        m = car.m

        # NOTE here vx = vf, vy = vs, different convention
        x, y, heading, vx, vy, omega = car_states
        steering, throttle = control
        d_vx, d_vy, omega = self.core_dynamics(
            (vx, vy, omega), control, car, dt)

        # discretization
        vx = vx + d_vx * dt
        vy = vy + d_vy * dt

        # back to global frame
        vxg = vx*cos(heading)-vy*sin(heading)
        vyg = vx*sin(heading)+vy*cos(heading)

        # update x,y, heading
        x += vxg*dt
        y += vyg*dt
        heading += omega*dt

        car_states = x, y, heading, vx, vy, omega
        return np.array(car_states)
