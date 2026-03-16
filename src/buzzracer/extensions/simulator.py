''' Base class for all simulators '''
from __future__ import annotations
from typing import TYPE_CHECKING
from time import time, sleep
from enum import Enum, unique
from abc import ABC, abstractmethod

import numpy as np

from buzzracer.common import ExperimentType
from buzzracer.extensions.extension import Extension
from buzzracer.sysid.vehicle_dynamics import VehicleDynamics
from buzzracer.types import CartesianState, CurvilinearState, Control
if TYPE_CHECKING:
    from buzzracer.cars.car import Car


@unique
class NoiseType(Enum):
    NORMAL = 1
    UNIFORM = 2
    IMPULSE = 3


class Simulator(Extension, ABC):
    '''
    Base class for simulators

    car.state = CartesianState(x,y,heading,v_forward,v_sideway,omega)
    however simulator can establish a property car.sim_state
    that use different state representation for simulation
    '''
    state_type: type[CartesianState] | type[CurvilinearState] = CartesianState
    ''' State type used by this simulator, default cartesian'''

    def __init__(self):
        super().__init__(handle_name='simulator')
        # self.print_debug_enable()
        self.match_time: bool = False
        ''' If True, attempt to match simulation with clock time. Pauses at each step.'''
        self.print_info('match_time: ' + str(self.match_time))
        self.dynamics_model: type[VehicleDynamics] = VehicleDynamics
        ''' Dynamics model to use for simulation, must be overridden in config
        possible values: KinematicBicycleModelFrenet, DynamicBicycleModelCartesian, etc.'''
        self.state_noise_enabled: bool = None
        ''' If True, enable state noise '''
        self.state_noise_magnitude: float = None
        ''' Noise magnitude, multipled to unit noise of the chosen noise type'''
        self.state_noise_type: NoiseType = None
        ''' Type of noise to add '''
        self.impulse_state_noise_probability: float = None
        ''' Probability of the impulse state noise being added,(0,1), only used for impulse noise'''
        self.cars: list[Car] = []
        ''' List of all cars using this simulator. This may be a subset of main.cars'''

        self.t0 = None
        self.real_sim_time_ratio = 1.0
        ''' Real time / sim time. If larger than 1.0, simulation will be slowed down.
            This allow easier human interpretation of fast simulations.
            Only useful if match_time = True '''
        self.print_info('real/sim time ratio = %.1f ' %
                        (self.real_sim_time_ratio))

        self.sim_t = 0
        ''' Elapsed time in simulation'''
        self.cars: Car = []

        if self.main.config.experiment_type != ExperimentType.Simulation:
            self.print_error(
                'Experiment type is not Simulation but a Simulator is loaded')

        if self.state_noise_enabled:
            assert self.state_noise_type is not None
            assert self.state_noise_magnitude is not None
            self.state_noise_magnitude = np.array(self.state_noise_magnitude)
            noise_type_to_fun = {NoiseType.UNIFORM: self.add_state_noise_uniform,
                                 NoiseType.NORMAL: self.add_state_noise_normal,
                                 NoiseType.IMPULSE: self.add_state_noise_impulse}
            self.add_state_noise = noise_type_to_fun[self.state_noise_type]

    def add_car(self, car):
        """register a car to use this simulation. """
        self.cars.append(car)

    # TODO use Replay.VehicleDynamics
    @staticmethod
    @abstractmethod
    def advance_dynamics(state: CurvilinearState | CartesianState,
                         control: Control,
                         car: Car,
                         dt: float,
                         curvature: float = None) -> CurvilinearState | CartesianState:
        """advance dynamics by dt.

        Args:
            state: state of the car, may be CartesianState or CurvilinearState
            control: (steering,throttle) steering in rad, left positive, throttle in [-1,1],
                    positive indicates acceleration
            car: Car object, contains information about the car's kinematics,
                also contains car.sim_state for simulators that do not use car.state for update
            dt: Time step to advance dynamics by, unit:seconds
            curvature: signed curvature
        Return:
            state at next time step.
        """
        return

    def update(self):
        for car in self.cars:
            if self.state_type == CartesianState:
                # NOTE cartesian state is passed directly as a tuple for now
                car.state = self.advance_dynamics(
                    car.state, (car.steering, car.throttle), car, self.main.config.dt)
            elif self.state_type == CurvilinearState:
                control = Control(steering=car.steering, throttle=car.throttle)
                curvature = self.main.track.curvature_s(car.sim_state.progress)
                car.sim_state = self.advance_dynamics(
                    car.sim_state, control, car, self.main.dt, curvature)
                car.state = self.main.track.curv_to_cart(car.sim_state)

        if self.state_noise_enabled:
            self.addStateNoise()
        self.main.new_state_update.set()
        self.sim_t += self.main.config.dt
        self.match_real_time()

    def match_real_time(self):
        """Sleep to match simulation time to clock time only works when then
        entire simulation loop runs faster than realtime."""
        if not self.match_time:
            return
        if self.t0 is None:
            self.t0 = time()
        time_to_reach = self.sim_t * self.real_sim_time_ratio + self.t0
        margin = time_to_reach - time()
        self.print_debug('sim_t = %.3f, world time = %.3f, target world time= %.3f, margin = %.3f' % (
            self.sim_t, time()-self.t0, self.sim_t*self.real_sim_time_ratio, margin))
        if margin < -1e-1:
            self.print_warning("Simulation loop can't keep up ..... lagging %.3f s" % (-margin))

        sleep(max(0, time_to_reach - time()))

    def add_state_noise_normal(self):
        for car in self.cars:
            car.state += np.random.normal(size=car.state.shape) * \
                self.state_noise_magnitude * self.main.dt

    def add_state_noise_uniform(self):
        for car in self.cars:
            car.state += np.random.uniform(low=-1.0, high=1.0, size=car.state.shape) * \
                self.state_noise_magnitude * self.main.dt

    def add_state_noise_impulse(self):
        for car in self.cars:
            val = np.random.uniform()
            if val < self.impulse_state_noise_probability:
                car.state += self.state_noise_magnitude * self.main.dt
