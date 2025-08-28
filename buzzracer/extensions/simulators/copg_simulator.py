''' Copg simulator, use curvilinear ref frame dynamics from Competitive Policy Gradient paper '''
# NOTE not maintained
# pylint: disable=all

from math import sin, cos

import numpy as np

from buzzracer.RL.copg.rcvip_simulator.VehicleModel import VehicleModel
from buzzracer.extensions.simulators.kinematic_simulator import KinematicSimulator
from buzzracer.extensions.simulator import Simulator


class CopgSimulator(Simulator):
    ''' Copg Simulator'''
    def __init__(self):
        super().__init__()
        self.cars = self.main.cars

    def init(self):
        super().init()

        CopgSimulator.dt = self.main.dt
        KinematicSimulator.dt = CopgSimulator.dt
        KinematicSimulator.max_v = 100
        for car in self.cars:
            self.add_car(car)
        self.main.new_state_update.set()

        CopgSimulator.vehicle_model = VehicleModel(
            1, 'cpu', 'rcp', dt=self.main.dt)

    # add a car to be DynamicSimu
    # car needs to (x,y,heading,v_forward,v_sideway,omega)
    def add_car(self, car):
        x, y, heading, v_forward, v_sideway, _ = car.states
        car.Vx = v_forward
        car.Vy = v_sideway

        car.x = x
        car.y = y
        car.psi = heading

        car.d_x = car.Vx*cos(car.psi)-car.Vy*sin(car.psi)
        car.d_y = car.Vx*sin(car.psi)+car.Vy*cos(car.psi)
        car.d_psi = 0
        car.sim_states = np.array(
            [car.x, car.d_x, car.y, car.d_y, car.psi, car.d_psi])

        car.state_dim = 6
        car.control_dim = 2

        # not implemented: support for artificially added noise
        noise = False
        car.noise = noise

        # car.states_hist = []
        car.local_states_hist = []
        car.norm = []

    @staticmethod
    def advance_dynamics(car_states, control, car, dt):
        """# advance vehicle dynamics.

        # NOTE using car frame origined at CG with x pointing forward, y leftward
        # this method does NOT update car.sim_states, only returns a sim_state
        # this is to make itself useful for when update is not necessary
        #    x,y,psi,v_forward,v_sideway,d_psi = car_states
        # x,y,psi,v_forward,v_sideway,d_psi = car_states
        # control = steering,throttle

        """
        del car
        del dt
        local_state = CopgSimulator.vehicle_model.from_global_to_local(car_states)
        new_local_state = CopgSimulator.vehicle_model.dyn_model_blend_batch(
            local_state, (control[1], control[0]))
        global_state = CopgSimulator.vehicle_model.from_local_to_global(
            new_local_state).flatten()
        return global_state.flatten()
