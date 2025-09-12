''' Learned model from Gaussian Process NOTE this has not been refactored'''
from __future__ import annotations
from typing import TYPE_CHECKING

import torch

from buzzracer.types import CartesianState, Control
from buzzracer.sysid.gaussian_process.gpModel import MultitaskDeepGP
from buzzracer.sysid.dynamic_bicycle_model import DynamicBicycleModelCartesian
if TYPE_CHECKING:
    from buzzracer.cars.car import Car


class GpModel(DynamicBicycleModelCartesian):
    ''' Learned Gaussian Process model based on Dynamic Bicycle Model'''

    def __init__(self):
        super().__init__()

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
