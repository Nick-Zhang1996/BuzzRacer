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

def getPerformanceMetric(p1_policy_pth):
    batch_size  = 10000
    curr_batch_size = batch_size

    p1 = Actor(5, 2, std=0.1)
    p1.load_state_dict(torch.load(p1_policy_pth))

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

    for i in range(2000):

        #sample action from random policy
        dist = p1(state[:,0:5])
        action = dist.sample().to('cpu')

        prev_state = state

        # advance state
        state = vehicle_model.dynModelBlendBatch(state, action)

        bounds = vehicle_model.getLocalBounds(state[:, 0])
        reward,  done = getRewardSingleAgent(state, bounds ,prev_state,  device)

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

        # TODO if an episode lasts longer than 2000
        if state.size(0)==0 or i==1999:
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

i = 3460
p1_policy_pth = f"./trained_model/gda_sample/model/agent1_{i}.pth"
print(f' evaluating policy pair {i}')
getPerformanceMetric(p1_policy_pth)
