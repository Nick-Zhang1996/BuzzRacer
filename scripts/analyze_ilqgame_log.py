from buzzracer.common import *
import os
import sys
import numpy as np
import pickle
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class Main(PrintObject):
    def __init__(self):
        self.basedir = os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))))

    def load_state_log(self, log_name):
        # time_steps * cars * (states + action)
        full_path = os.path.join(self.basedir, 'log', log_name)
        self.print_ok(f'opening file at {full_path}')
        self.skip = 0
        with open(full_path, 'rb') as f:
            self.data = np.array(pickle.load(f))
        self.data = self.data[self.skip:]
        self.data[:, :, 0] -= self.data[0, 0, 0]
        # create cars
        self.car_count = self.data.shape[1]
        return self.data

    def load_debug_dict(self, log_name):
        # time_steps * cars * (states + action)
        full_path = os.path.join(self.basedir, 'log', log_name)
        self.print_ok(f'opening file at {full_path}')
        with open(full_path, 'rb') as f:
            data = pickle.load(f)
        return data


if __name__ == '__main__':
    main = Main()
    # dim: time,car, state
    # (time, x,y,theta,vforward,vsideway=0,omega)
    # data = main.load_log('2023_11_13_exp/full_state2.p')
    data = main.load_debug_dict('2023_11_15_exp/debug_dict1.p')
    breakpoint()
    print('done')
