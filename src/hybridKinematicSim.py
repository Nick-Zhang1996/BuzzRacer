# hybrid simulator with kinematic base model
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

from math import cos,sin
# rewrite advCarSim.py as pytorch nn module

class hybridKinematicSim(nn.Module):

    def __init__(self, dtype, device, history_steps, forward_steps, dt):
        torch.autograd.set_detect_anomaly(True)
        super(hybridKinematicSim,self).__init__()
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

        self.lf = self.get_param(0.09-0.036,False)
        self.lr = self.get_param(0.036,False)

        #self.throttle_offset = self.get_param(0.26,True)
        #self.throttle_ratio = self.get_param(7.003,True)
        self.throttle_offset = self.get_param(0.24,False)
        self.throttle_ratio = self.get_param(6.98,False)

        # heuristic functions
        self.understeer_coeff = self.get_param(0.68,True)

        # residual neural network
        # input:
        # - local dynamic states (Vx,Vy,omega, throttle, steering)
        # NOTE are they on the same order?
        # output:
        # - residual longitudinal acc (scalar)
        # - residual lateral acc (scalar)
        # - residual angular acc (scalar)
        hidden_layer_depth = 16
        self.fc1 = nn.Linear(history_steps * 5, hidden_layer_depth)
        self.fc2 = nn.Linear(hidden_layer_depth, 3)
        if dtype == torch.float:
            self.fc1 = self.fc1.float()
            self.fc2 = self.fc2.float()
        elif dtype == torch.double:
            self.fc1 = self.fc1.double()
            self.fc2 = self.fc2.double()
        self.residual_bounds = self.get_tensor([5.0,5.0,3.0],False)

    def get_residual_acc(self,dynamic_states):
        batch_size, history_steps, state_dim = dynamic_states.size()
        y1 = torch.tanh(self.fc1(dynamic_states.view(batch_size, -1)))
        y2 = torch.tanh(self.fc2(y1)) * self.residual_bounds.view(1, 3)
        residual_longitudinal_acc = y2[:, 0]
        residual_lateral_acc = y2[:, 1]
        residual_torque = y2[:,2]
        return residual_longitudinal_acc,residual_lateral_acc,residual_torque

    # predict future car state
    # Input: 
    #  (tensors)
    #       states: batch_size * history_steps * full_state(states+commands)
    #       actions: batch_size * forward_steps * actions(throttle,steering)
    #   concatenated
    # Output:
    #       future_states: forward_steps * (states)
    # NOTE NOTE an inconsistency here, the model assumes CG of the car to be car frame origin while the data is using center of rear axle
    def forward(self,full_states,actions, enable_rnn):
        assert len(full_states.size()) == 3
        batch_size, history_steps, full_state_dim = full_states.size()
        assert history_steps == self.history_steps
        assert full_state_dim == (self.state_dim + self.action_dim)

        # base simulation, kinematic model

        for i in range(self.forward_steps):
            latest_state = full_states[:,-1,-(self.state_dim+self.action_dim):-self.action_dim]
            latest_action = full_states[:,-1,-self.action_dim:]
            throttle = latest_action[:,0]
            steering = latest_action[:,1]
            # change ref frame to car frame(forward +x, left +y)
            psi = latest_state[:,4]
            # vehicle forward speed
            Vx = latest_state[:,1]*torch.cos(psi) + latest_state[:,3]*torch.sin(psi)
            long_acc = self.getLongitudinalAcc(throttle)
            #u = torch.cat((long_acc.unsqueeze(1),steering.unsqueeze(1)),dim=1)

            # self.states: x,vx,y,vy,psi,dpsi
            # state derivative in local frame
            R = (self.lr+self.lf)/torch.tan(steering)
            beta = torch.atan(self.lr/R)

            Vx = Vx + long_acc*self.dt
            Vy = (latest_state[:,1]**2 + latest_state[:,3]**2)**0.5 * torch.sin(beta)
            omega = Vx/R


            if enable_rnn:
                # assemble input for get_residual_acc
                # input: batch_size * history_steps * (vx,vy(local), omega, throttle,steering)
                psi_hist = full_states[:,-self.history_steps:,4]
                Vx_hist = full_states[:,-self.history_steps:,1]*torch.cos(psi_hist) + full_states[:,-self.history_steps:,3]*torch.sin(psi_hist)
                Vy_hist = -full_states[:,-self.history_steps:,1]*torch.sin(psi_hist) + full_states[:,-self.history_steps:,3]*torch.cos(psi_hist)
                omega_hist = full_states[:,-self.history_steps:,5]
                throttle_hist = full_states[:,-self.history_steps:,6]
                steering_hist = full_states[:,-self.history_steps:,7]

                '''
                Vx_hist = Vx_hist.unsqueeze(2)
                Vy_hist = Vy_hist.unsqueeze(2)
                omega_hist = omega_hist.unsqueeze(2)
                throttle_hist = throttle_hist.unsqueeze(2)
                steering_hist = steering_hist.unsqueeze(2)
                '''

                local_dynamic_state_hist = torch.stack([Vx_hist,Vy_hist,omega_hist,throttle_hist,steering_hist],dim=1)

                residual_longitudinal_acc,residual_lateral_acc,residual_angular_acc = self.get_residual_acc(local_dynamic_state_hist.squeeze(1))
                Vx = Vx + residual_longitudinal_acc * self.dt
                Vy = Vy + residual_lateral_acc * self.dt
                omega = omega + residual_angular_acc * self.dt

            # NOTE heuristic function
            Vy = Vy - self.understeer_coeff * Vx * steering

            # back to global frame
            Vxg = Vx*torch.cos(psi) - Vy*torch.sin(psi)
            Vyg = Vx*torch.sin(psi) + Vy*torch.cos(psi)

            # advance model
            predicted_state = latest_state.clone()
            # x
            predicted_state[:,0] = predicted_state[:,0] + Vxg * self.dt
            predicted_state[:,1] = Vxg
            predicted_state[:,2] = predicted_state[:,2] + Vyg * self.dt
            predicted_state[:,3] = Vyg
            predicted_state[:,4] = predicted_state[:,4] + omega * self.dt
            predicted_state[:,5] = omega

            # angle wrapping
            predicted_state[:,4] = (predicted_state[:,4]+np.pi)%(2*np.pi)-np.pi
            new_full_state = torch.cat([predicted_state.view(batch_size,1,self.state_dim), actions[:,i,:].view(batch_size,1,self.action_dim)],dim=2)
            full_states = torch.cat([full_states, new_full_state],dim=1)

        future_states = full_states[:,-self.forward_steps:,:]
        # FIXME
        # verify this function against authentic simulation
        '''
        errors = []
        for i in range(batch_size):
            last_full_state = np.array(full_states[i,-self.forward_steps-1,:].detach())
            next_full_state = np.array(future_states[i,0,:].detach())
            advsim_next_full_state = self.testForward(full_states,actions,i)
            # this should be close to zero
            error = np.linalg.norm(next_full_state-advsim_next_full_state)
            errors.append(error)
            if error>1e-10:
                print("last_state")
                print(last_full_state)
                print("hybrid sim")
                print(next_full_state)
                print("advSim")
                print(advsim_next_full_state)
                print("error norm")
                print(error)
                print("----")
        print(np.mean(errors))
        '''
        return future_states

    # using advCarSim, calculate future states
    def testForward(self,full_states, actions,i=0,sim = None):
        # only do the first one
        latest_state = full_states[i,-1,-(self.state_dim+self.action_dim):-self.action_dim]
        latest_action = full_states[i,-1,-self.action_dim:]
        # change ref frame to car frame
        latest_state = latest_state.detach().numpy()
        latest_action = latest_action.detach().numpy()
        psi = latest_state[4]
        Vx = latest_state[1]*cos(psi) + latest_state[3]*sin(psi)
        #print("test_forward")
        #print(latest_state)

        # current parameters in hybrid simulator
        '''
        Caf = self.Caf.detach().item()
        Car = self.Car.detach().item()
        lf = self.lf.detach().item()
        lr = self.lr.detach().item()
        m = self.m.detach().item()
        Iz = self.Iz.detach().item()
        '''
        Caf = sim.Caf
        Car = sim.Car
        lf = sim.lf
        lr = sim.lr
        m = sim.m
        Iz = sim.Iz

        A = np.array([[0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0],
                    [0, 0, 0, -(2*Caf+2*Car)/(m*Vx), 0, -Vx-(2*Caf*lf-2*Car*lr)/(m*Vx)],
                    [0, 0, 0, 0, 0, 1],
                    [0, 0, 0, -(2*lf*Caf-2*lr*Car)/(Iz*Vx), 0, -(2*lf**2*Caf+2*lr**2*Car)/(Iz*Vx)]])
        #print("testForward A")
        #print(A)
        B = np.array([[0,1,0,0,0,0],[0,0,0,2*Caf/m,0,2*lf*Caf/Iz]]).T
        u = latest_action

        # active roattion matrix of angle(rad)
        R = lambda angle: np.array([[cos(angle), 0,-sin(angle),0,0,0],
                        [0, cos(angle), 0,-sin(angle),0,0],
                        [sin(angle),0,cos(angle),0,0,0],
                        [0,sin(angle),0,cos(angle),0,0],
                        [0,0,0,0,1,0],
                        [0,0,0,0,0,1]])
        states = latest_state
        states = states + R(psi) @ (A @ R(-psi) @ states + B @ u)*self.dt
        actions = actions.detach().numpy()
        predicted_full_states = np.hstack([states,actions[i,0,:]])
        return predicted_full_states


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
        return nn.Parameter(self.get_tensor([init_val],requires_grad),requires_grad)

    # get Caf
    def Caf(self):
        ratio = (torch.tanh(self.Caf_param) + 1)/2.0 * (self.Caf_range[1] - self.Caf_range[0]) + self.Caf_range[0]
        return self.Caf_base * ratio

    def Car(self):
        return self.Caf()

    def Iz(self):
        pow_ratio = (torch.tanh(self.Iz_param) + 1)/2.0 * (self.Iz_pow_ratio[1] - self.Iz_pow_ratio[0]) + self.Iz_pow_ratio[0]
        return self.Iz_base * torch.pow(10.0,pow_ratio)

    def getLongitudinalAcc(self,throttle):
        acc = (throttle - self.throttle_offset) * self.throttle_ratio
        return acc



# simple test
if __name__ == '__main__':
    log_names =  glob.glob('../log/sysid/full_state*.p')
    dt = 0.01
    history_steps = 5
    forward_steps = 5
    dtype = torch.double
    device = torch.device('cpu')


    criterion = nn.MSELoss()

    simulator = hybridKinematicSim(dtype, device, history_steps, forward_steps, dt)
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
        predicted_state = simulator(full_states,actions,True)

        target_states = batch[:,-forward_steps:,:]

        loss = criterion((predicted_state - full_states_mean) / full_states_std, (target_states - full_states_mean) / full_states_std)
        epoch_loss = epoch_loss + loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(epoch_loss.detach().item())

