import sys
import os
import glob

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import scipy.optimize

from common import ndarray

from torch.utils.data import DataLoader
from sysidDataloader import CarDataset
# rewrite advCarSim.py as pytorch nn module

class hybridSim(nn.Module):

    def __init__(self, dtype, device, history_steps, forward_steps, dt):
        super(hybridSim,self).__init__()
        assert dtype == torch.float or dtype == torch.double
        assert history_steps >= 1 and forward_steps >= 1
        assert 1e-3 <= dt <= 2e-2

        self.state_dim = 6
        self.action_dim = 2

        self.dtype = dtype
        self.device = device
        self.history_steps = history_steps
        self.forward_steps = forward_steps
        self.dt = dt

        self.g = 9.81
        m = 0.2
        self.m = self.get_param(m)
        self.Caf = self.get_param(5*0.25*m*self.g)
        self.Car = self.get_param(5*0.25*m*self.g)
        self.lf = self.get_param(0.05)
        self.lr = self.get_param(0.05)
        # approximate as a solid box
        self.Iz = self.get_param(1/12.0*(0.1**2+0.1**2))


    # predict future car state
    # Input: 
    #  (tensors)
    #       states: batch_size * history_steps * (states+commands)
    #       actions: batch_size * forward_steps * actions(throttle,steering)
    #   concatenated
    # Output:
    #       future_states: forward_steps * (states)
    def forward(self,full_states,actions):
        assert len(full_states.size()) == 3
        batch_size, history_steps, full_state_dim = full_states.size()
        assert history_steps == self.history_steps
        assert full_state_dim == (self.state_dim + self.action_dim)

        # base simulation, dynamic model

        for i in range(self.forward_steps):
            latest_state = full_states[:,-1,-(self.state_dim+self.action_dim):-self.action_dim]
            latest_action = full_states[:,-1,-self.action_dim:]
            # change ref frame to car frame
            psi = latest_state[:,4]
            Vx = latest_state[:,1]*torch.cos(psi) + latest_state[:,3]*torch.sin(psi)

            # TODO maybe doesnt need grad
            A = torch.zeros((batch_size,6,6),dtype=self.dtype,requires_grad=False)
            A[:,0,1] = 1
            A[:,2,3] = 1
            A[:,3,3] = -(2*self.Caf+2*self.Car)/(self.m*Vx)
            A[:,3,5] = -Vx-(2*self.Caf*self.lf-2*self.Car*self.lr)/(self.m*Vx)
            A[:,4,5] = 1
            A[:,5,3] = -(2*self.lf*self.Caf-2*self.lr*self.Car)/(self.Iz*Vx)
            A[:,5,4] = -(2*self.lf**2*self.Caf+2*self.lr**2*self.Car)/(self.Iz*Vx)

            B = torch.zeros((batch_size,6,2),dtype=self.dtype,requires_grad=False)
            B[:,1,0] = 1
            B[:,3,1] = 2*self.Caf/self.m
            B[:,5,1] = 2*self.lf*self.Caf/self.Iz

            u = latest_action
            #self.states = self.states + R(psi) @ (A @ R(-psi) @ self.states + B @ u)*dt
            temp = torch.matmul(A,torch.matmul(self.get_R(-psi),latest_state.unsqueeze(2)))+torch.matmul(B,u.unsqueeze(2))
            predicted_state = latest_state.unsqueeze(2) +  torch.matmul(self.get_R(psi),temp)*self.dt
            new_full_state = torch.cat([predicted_state.view(batch_size,1,self.state_dim), actions[:,i,:].view(batch_size,1,self.action_dim)],dim=2)
            full_states = torch.cat([full_states, new_full_state],dim=1)


        future_states = full_states[:,-self.forward_steps:,:]
        return future_states

    # get active rotation matrix, batch style
    def get_R(self,psi):
        batch = psi.size()[0]
        R = torch.zeros((batch,6,6),dtype=self.dtype,requires_grad=False)
        R[:,0,0] = torch.cos(psi)
        R[:,0,2] = -torch.sin(psi)
        R[:,1,1] = torch.cos(psi)
        R[:,1,3] = -torch.sin(psi)
        
        R[:,2,0] = torch.sin(psi)
        R[:,2,2] = torch.cos(psi)
        R[:,3,1] = torch.sin(psi)
        R[:,3,3] = torch.cos(psi)

        R[:,4,4] = 1
        R[:,4,4] = 1
        R[:,4,4] = 1
        R[:,4,4] = 1
        R[:,5,5] = 1
        return R


    def get_tensor(self, init_val, requires_grad):
        return torch.tensor(ndarray(init_val), dtype=self.dtype, device=self.device, requires_grad=requires_grad)
    def get_param(self,init_val,requires_grad=True):
        return nn.Parameter(self.get_tensor([init_val],True))


# simple test
if __name__ == '__main__':
    log_names =  glob.glob('../log/sysid/full_state*.p')
    dt = 0.01
    history_steps = 5
    forward_steps = 5
    dtype = torch.double
    device = torch.device('cpu')


    criterion = nn.MSELoss()

    simulator = hybridSim(dtype, device, history_steps, forward_steps, dt)
    simulator.to(device)
    optimizer = optim.Adam(simulator.parameters(), lr=1e-3)

    dataset = CarDataset(log_names,dt,history_steps,forward_steps)
    data_loader = DataLoader(dataset, batch_size=512, shuffle=True, num_workers=4)
    full_states_mean = torch.tensor(dataset.full_states_mean, dtype=dtype, device=device, requires_grad=False).view(1, simulator.state_dim+simulator.action_dim)
    full_states_std = torch.tensor(dataset.full_states_std, dtype=dtype, device=device, requires_grad=False).view(1, simulator.state_dim+simulator.action_dim)

    epoch_loss = 0
    for i, batch in enumerate(data_loader):
        full_states = batch[:,:history_steps,:]
        actions = batch[:,-forward_steps:,-simulator.action_dim:]
        full_states = full_states.to(device)
        actions = actions.to(device)
        predicted_state = simulator(full_states,actions)

        target_states = batch[:,-forward_steps:,:]

        loss = criterion((predicted_state - full_states_mean) / full_states_std, (target_states - full_states_mean) / full_states_std)
        epoch_loss += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

