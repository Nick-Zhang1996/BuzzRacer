import sys
import os

import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import glob
import pickle
from scipy.signal import savgol_filter
from common import *

class CarDataset(Dataset):

    def __init__(self, log_names, dt, history_steps, forward_steps):
        self.state_dim = 6
        self.action_dim = 2

        self.raw_data = []
        # Compute the mean and std of full states(state+action).
        full_states_sum = 0
        full_states_sq_sum = 0
        full_states_cnt = 0
        for filename in log_names:
            with open(filename, 'rb') as f:
                data = pickle.load(f)
                data = np.array(data)
                t = data[:,0]
                t = t-t[0]
                x = data[:,1]
                y = data[:,2]
                heading = data[:,3]
                steering = data[:,4]
                throttle = data[:,5]
                dx = np.diff(x)
                dy = np.diff(y)
                dpsi = np.diff(heading)

                vx = savgol_filter(dx/dt,51,2)
                vy = savgol_filter(dy/dt,51,2)
                omega = savgol_filter(dpsi/dt,51,2)

                data_segment = np.array([x[:-1],vx,y[:-1],vy,heading[:-1],omega,throttle[:-1],steering[:-1]]).T
                self.raw_data.append(data_segment)

        full_states = np.vstack(self.raw_data)
        full_states_sum += np.sum(full_states, axis=0)
        full_states_sq_sum += np.sum(full_states ** 2, axis=0)
        full_states_cnt += full_states.shape[0]

        self.full_states_mean = full_states_sum / full_states_cnt
        self.full_states_std = np.sqrt(full_states_sq_sum / full_states_cnt - self.full_states_mean ** 2)

        # TODO select good data section (no hitting, no reverse etc)
        self.dataset = []
        for segment in self.raw_data:
            for i in range(len(segment)-history_steps-forward_steps+1):
                datum = segment[i:i+history_steps+forward_steps,:]
                self.dataset.append(np.array(datum))
        self.dataset = np.array(self.dataset)
        print_info("dataset size:"+str(self.dataset.shape))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def plotDataset(self):
        return

if __name__ == '__main__':
    log_names =  glob.glob('../log/sysid/full_state*.p')
    print(log_names)
    CarDataset(log_names,0.01,5,5)
