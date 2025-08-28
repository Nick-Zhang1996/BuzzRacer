''' Replay state history '''

import os
import pickle
from math import sin, cos, tan, atan
from typing import NamedTuple

import torch
import numpy as np
from scipy.interpolate import splev

from buzzracer.common import BASEDIR
from buzzracer.util.timeUtil import ExecutionTimer
from buzzracer.extensions.simulator.dynamic_simulator import DynamicSimulator
from buzzracer.extensions.Simulator import Simulator
from buzzracer.sysid.tire import tire_curve
from buzzracer.sysid.gaussian_process.gpModel import MultitaskDeepGP
from buzzracer.car.car import Car


class CartesianState(NamedTuple):
    x: float
    y: float
    heading: float
    v_forward: float
    ''' Longitudinal speed, forward positive '''
    v_sideway: float
    ''' Lateral speed, left positive '''
    omega: float


class CurvilinearState(NamedTuple):
    progress: float
    lateral_err: float
    rel_heading: float
    ''' Relative heading w.r.t. reference curve'''
    v_forward: float
    ''' Longitudinal speed, forward positive '''
    v_sideway: float
    ''' Lateral speed, left positive '''
    omega: float
    ''' Relative heading time rate w.r.t. reference curve'''


class Control(NamedTuple):
    steering: float
    ''' Steering angle in rad, left positive '''
    throttle: float
    ''' Throttle, positive forward, negative braking '''


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
        heading = state.rel_heading + track_heading
        # NOTE we ignored second order curvature in reference curve
        return CartesianState(x=x,
                              y=y,
                              heading=heading,
                              v_forward=state.v_forward,
                              v_sideway=state.v_sideway,
                              omega=state.omega)

    def update(self):
        if self.timestep >= self.data.shape[0]:
            self.main.exit_request.set()
            return

        if self.curvilinear:
            for (i, car) in enumerate(self.main.cars):
                car.states = self.CurvilinearToCartesian(
                    self.data[self.timestep, i])
        else:
            for (i, car) in enumerate(self.main.cars):
                car.states = tuple(self.data[self.timestep, i, 1:7].flatten())
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


# TODO make this a project wide dependency
class VehicleDynamics:
    """ Base class for vehicle dynamics model."""

    def __init__(self):
        self.curvilinear = None
        ''' If True, then use CurvilinearState, else use CartesianState'''

    def advance_dynamics(self, state: CartesianState | CurvilinearState,
                         control: Control, car: Car,
                         dt: float) -> CartesianState | CurvilinearState:
        ''' Step dynamics forward by dt, x+ = x + f(x,u)*dt

        Args:
            state: differs depending on self.cartesian, 
            control: steering,throttle
            car: Car object to supply vehicle parameters
            dt: time step in seconds e.g. 0.01
        Return:
            states at next timestep
        '''
        del control
        del car
        del dt
        return state


class DynamicBicycleModel(VehicleDynamics):
    ''' Dynamic bicycle model with pacjka tire model'''

    def __init__(self):
        super().__init__()
        self.curvilinear = False

    def core_dynamics(self, core_state: tuple[float], control: Control,
                      car: Car, dt: float) -> tuple[float]:
        '''
        Calculate dynamic bicycle model state time derivative. 

        Args: 
            core_state: (vx,vy,omega) 
            control: (steering,throttle)
            car: Car object to provide parameters
            dt: time step size in seconds
        Return: 
            State derivatives (d_vx, d_vy, d_omega)
        '''
        lf = car.lf
        lr = car.lr
        L = car.L

        Iz = car.Iz
        m = car.m
        vx, vy, omega = core_state

        # for small longitudinal velocity use kinematic model
        if vx < 0.05:
            beta = atan(lr / L * tan(control.steering))

            def norm(a, b):
                return (a**2 + b**2)**0.5

            # motor model
            d_vx = 6.17 * (control.throttle - vx / 15.2 - 0.333)
            d_vy = (norm(vx, vy) * sin(beta) - vy) / dt
            d_omega = (vx / L * tan(control.steering) - omega) / dt

        else:
            slip_f = -np.arctan((omega * lf + vy) / vx) + control.steering
            slip_r = np.arctan((omega * lr - vy) / vx)

            # Ffy = Df * np.sin( C * np.arctan(B *slip_f)) * 9.8 * lr / (lr + lf) * m
            # Fry = Dr * np.sin( C * np.arctan(B *slip_r)) * 9.8 * lf / (lr + lf) * m
            Ffy = tire_curve(slip_f) * m * 9.8 * lr / (lr + lf)
            Fry = 1.15 * tire_curve(slip_r) * m * 9.8 * lf / (lr + lf)

            # Dynamics
            # d_vx = 1.0/m * (Frx - Ffy * np.sin( control.steering ) + m * vy * omega)
            d_vx = 6.17 * (control.throttle - vx / 15.2 - 0.333)
            d_vy = 1.0 / m * (Fry + Ffy * np.cos(control.steering) -
                              m * vx * omega)
            d_omega = 1.0 / Iz * (Ffy * lf * np.cos(control.steering) -
                                  Fry * lr)
        return (d_vx, d_vy, d_omega)

    def advance_dynamics(self, state: CartesianState, control: Control,
                         car: Car, dt: float) -> CartesianState:
        ''' Step dynamics forward by dt, x+ = x + f(x,u)*dt

        Args:
            state: differs depending on self.cartesian, 
            control: steering,throttle
            car: Car object to supply vehicle parameters
            dt: time step in seconds e.g. 0.01
        Return:
            state at next timestep

        This method uses a car frame origined at CG with x
        '''
        d_vx, d_vy, d_omega = self.core_dynamics(
            (state.v_forward, state.v_sideway, state.omega), control, car, dt)

        # Discretization
        vx = state.v_forward + d_vx * dt
        vy = state.v_sideway + d_vy * dt
        omega = state.omega + d_omega * dt

        # Back to global frame
        vxg = vx * cos(state.heading) - vy * sin(state.heading)
        vyg = vx * sin(state.heading) + vy * cos(state.heading)

        # Update x,y, heading
        x = state.x + vxg * dt
        y = state.y + vyg * dt
        heading = state.heading + omega * dt + 0.5 * d_omega * dt * dt

        return CartesianState(x=x,
                              y=y,
                              heading=heading,
                              v_forward=vx,
                              v_sideway=vy,
                              omega=omega)


class KinematicBicycleModel(VehicleDynamics):
    ''' Kinematic Bicycle Model '''

    def __init__(self):
        super().__init__()
        self.curvilinear = False

    def advance_dynamics(self, state: CartesianState, control: Control,
                         car: Car, dt: float) -> CartesianState:
        ''' Step dynamics forward by dt, x+ = x + f(x,u)*dt

        Args:
            state: differs depending on self.cartesian, 
            control: steering,throttle
            car: Car object to supply vehicle parameters
            dt: time step in seconds e.g. 0.01
        Return:
            state at next timestep

        '''

        beta = np.arctan(np.tan(control.steering) * car.lr / (car.lf + car.lr))
        dxdt = state.v_forward * cos(state.heading + beta)
        dydt = state.v_sideway * sin(state.heading + beta)
        dvdt = 6.17 * (control.throttle - state.v_forward / 15.2 - 0.333)
        omega = dheadingdt = state.forward_v / car.lr * np.sin(beta)

        x = state.x + dt * dxdt
        y = state.y + dt * dydt
        v = state.v_forward + dt * dvdt
        heading = state.heading + dt * dheadingdt
        return CartesianState(x=x,
                              y=y,
                              heading=heading,
                              v_forward=v,
                              v_sideway=0,
                              omega=omega)


class GpModel(DynamicBicycleModel):
    ''' Learned Gaussian Process model based on Dynamic Bicycle Model'''

    def __init__(self):
        super().__init__()
        self.curvilinear = False

        # should move to config, but again this is pretty one-off
        model_filename = '/home/nickzhang/rcvip/src/sysid/gaussian_process/model.p'
        input_dim = 5
        output_dim = 3
        self.model = MultitaskDeepGP((100, input_dim), output_dim)
        self.model.load_state_dict(torch.load(model_filename))

    def core_dynamics(self, core_state: tuple[float], control: Control,
                      car: Car, dt: float) -> tuple[float]:
        '''
        Calculate dynamic bicycle model state time derivative. 

        Args: 
            core_state: (vx,vy,omega) 
            control: (steering,throttle)
            car: Car object to provide parameters
            dt: time step size in seconds
        Return: 
            State derivatives (d_vx, d_vy, d_omega)
        '''
        lr = car.lr
        L = car.L
        vx, vy, omega = core_state
        steering, throttle = control

        # for small longitudinal velocity use kinematic model
        if vx < 0.05:
            beta = atan(lr / L * tan(steering))

            def norm(a, b):
                return (a**2 + b**2)**0.5

            # motor model
            d_vx = 6.17 * (throttle - vx / 15.2 - 0.333)
            d_vy = (norm(vx, vy) * sin(beta) - vy) / dt
            # d_omega =
            omega = vx / L * tan(steering)

        else:
            model_input = torch.Tensor(core_state +
                                       tuple(control)).unsqueeze(0)
            mean, _ = self.model.predict(model_input)
            output = mean.numpy()
            d_vx = output[0, 0]
            d_vy = output[0, 1]
            omega = output[0, 2]
        return (d_vx, d_vy, omega)
