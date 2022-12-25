import sys
import numpy as np
import torch
import os
from common import *
from controller.CarController import CarController
from rl.copg.car_racing.network import Actor as Actor
from scipy.interpolate import splev as splev
from scipy.optimize import minimize as minimize
from time import time

class CopgCarController(CarController):
    def __init__(self, car,config):
        self.model_weights_pth = None
        super().__init__(car,config)
        self.track = self.main.track

        self.model = Actor(10,2, std=0.1)
        try:
            self.model.load_state_dict(torch.load(os.path.join(self.main.basedir, self.model_weights_pth)))
        except FileNotFoundError:
            self.print_error(f'cannot find specified .pth file')
            sys.exit(1)

    def init(self):
        super().init()
        for car in self.main.cars:
            car.last_s = 0.0

    def control(self):
        opponent = None
        # pick the first opponent
        for car in self.main.cars:
            if (car == self.car):
                continue
            opponent = car

        opponent_curvi_state = self.cartesianToCurvilinear(opponent)
        ego_curvi_state = self.cartesianToCurvilinear(self.car)

        opponent_curvi_state_torch = torch.from_numpy(np.array(opponent_curvi_state)).type(torch.FloatTensor)
        ego_curvi_state_torch = torch.from_numpy(np.array(ego_curvi_state)).type(torch.FloatTensor)

        # as in original model, last state angular velocity is ignored
        action_distribution = self.model(torch.cat([ego_curvi_state_torch[:5],opponent_curvi_state_torch[:5]]))
        action = action_distribution.sample().numpy()

        self.car.throttle = action[0]
        self.car.steering = action[1]

    # cart_state: x,y,heading,v_forward,v_sideways,omega
    # curvi_state: s,d,rel_heading,v_forward,v_sideways,omega
    def cartesianToCurvilinearFast(self,car):
        # every inquiry
        cart_state = car.states
        coord = (cart_state[0], cart_state[1])
        dist = np.sum((self.track.r-coord)**2, axis=1)
        idx = np.argmin(dist)
        s = self.track.s_vec[idx]
        r = self.track.r[idx]
        rp = coord - r
        dr = self.track.dr[idx]
        d = float(np.cross(dr,rp))
        rel_heading = cart_state[2] - np.arctan2(self.track.dr[idx,1],self.track.dr[idx,0])
        return (s,d,rel_heading,cart_state[3],cart_state[4],cart_state[5])

    def cartesianToCurvilinear(self,car):
        cart_state = car.states
        coord = np.array((cart_state[0], cart_state[1]))
        def dist_fun(u):
            disp = np.array(splev(u.item()%self.track.raceline_len_m, self.track.raceline_s,der=0)) - coord
            return disp[0]**2+disp[1]**2
        retval = minimize(dist_fun,car.last_s)
        s = retval.x.item()
        car.last_s = s

        r = np.array(splev(s%self.track.raceline_len_m, self.track.raceline_s,der=0)).flatten()
        rp = coord - r
        dr = np.array(splev(s%self.track.raceline_len_m, self.track.raceline_s,der=1)).flatten()
        d = float(np.cross(dr,rp)/np.linalg.norm(dr))
        rel_heading = cart_state[2] - np.arctan2(dr[1],dr[0])

        return (s,d,rel_heading,cart_state[3],cart_state[4],cart_state[5])

