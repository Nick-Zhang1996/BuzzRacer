import torch
import sys
import os
# sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0,'..') # inorder to run within the folder
sys.path.insert(0,'../..') # inorder to run within the folder
import numpy as np
import json
import pickle

from car_racing.network import Actor as Actor
from rcvip_env_function import getNFcollosionreward
import rcvip_simulator.VehicleModel as VehicleModel
from rcvip_env_function import getRewardSingleAgent

n_action = 2
n_batch = 2
n_pred = 200
n_state = 6

def getPerformanceMetric():
    batch_size  = 100
    curr_batch_size = batch_size

    device = torch.device("cpu")

    vehicle_model = VehicleModel.VehicleModel(n_batch, device, track='rcp')

    #  dim: batch, time, dimension
    mat_state = torch.empty(curr_batch_size,0,n_state)
    mat_action = torch.empty(curr_batch_size,0,n_action)
    mat_reward = torch.empty(curr_batch_size,0,1)
    mat_done = torch.empty(curr_batch_size,0,1)

    # dim: concatenated_time, dimension
    state_vec = torch.empty(0,n_state)
    action_vec = torch.empty(0,n_action)
    reward_vec = torch.empty(0,1)
    # torch.bool
    done_vec = torch.empty(0,1)

    # initial state
    state = torch.zeros(curr_batch_size, n_state)
    state[:,1] = (torch.rand(curr_batch_size) - 0.5) /0.5 * 0.1

    max_steps = 400
    for i in range(max_steps):
        #print(f'step {i}, remaining {state.size(0)}')
        action = torch.zeros(curr_batch_size,n_action)
        action[:,0] = 0.35
        action[:,1] = -state[:,1]*5

        prev_state = state

        # advance state
        state = vehicle_model.dynModelBlendBatch(state, action)

        bounds = vehicle_model.getLocalBounds(state[:, 0])
        reward,  done = getRewardSingleAgent(state, bounds ,prev_state,  device)

        # force terminate all episodes
        if (i==max_steps-1):
            done[:] = True

        # dim: batch,time,dim
        mat_state = torch.cat([mat_state, state.view(-1,1,n_state)], dim=1)
        mat_action = torch.cat([mat_action, action.view(-1,1,n_action)], dim=1)
        mat_reward = torch.cat([mat_reward,reward.view(-1,1,1)],dim=1)
        mat_done = torch.cat([mat_done, done.view(-1,1,1)], dim=1)


        if (np.any(done.numpy())):
            done_vec = torch.cat([done_vec, mat_done[done].view(-1,1)])
            reward_vec = torch.cat([reward_vec, mat_reward[done].view(-1,1)])
            state_vec = torch.cat([state_vec, mat_state[done].view(-1,n_state)])
            action_vec = torch.cat([action_vec, mat_action[done].view(-1,n_action)])


        # reduce state matrix
        state = state[~done]
        mat_done = mat_done[~done]
        mat_state = mat_state[~done]
        mat_reward = mat_reward[~done]
        mat_action = mat_action[~done]
        curr_batch_size = state.size(0)

        # all episodes finished
        if state.size(0)==0:
            break



    # total reward
    total_reward = torch.sum(reward_vec).numpy()
    total_episode = torch.sum(done_vec).numpy()
    episode_reward_mean = total_reward/total_episode

    # mean steps
    total_steps = state_vec.size(0)
    episode_steps_mean = total_steps / total_episode

    # mean reward
    step_reward_mean = total_reward / total_steps
    print(f' episode reward mean: {episode_reward_mean}')
    print(f' step reward mean: {step_reward_mean}')
    print(f' steps mean: {episode_steps_mean}')

    return 

getPerformanceMetric()
