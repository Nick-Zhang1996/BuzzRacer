# Game imports
import sys
import json

if (len(sys.argv) == 2):
    arg = sys.argv[1]
    if (".json" in arg):
        with open(arg,'r') as f:
            param = json.load(f)
        critic_lr = param['critic_lr']
        actor_lr = param['actor_lr']
        num_episode = param['num_episode']
        experiment_name = param['experiment_name']
        original_batch_size = batch_size = param['batch_size']
        num_episode = param['num_episode']
        print(f'Loading config from {arg}')
    else:
        critic_lr = 0.0005 # default 0.008
        actor_lr = 1e-5 # default 3e-5
        experiment_name = sys.argv[1]
        original_batch_size = batch_size = 32
        num_episode = 3000
        print(f'Using hard-coded params')
else:
    print(f'usage1: python gda_rcvip.py exp_name')
    print(f'usage2: python gda_rcvip.py param_json_name')
    exit(1)


import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, '..')
sys.path.insert(0,'../..')
from copg_optim import RCoPG as CoPG
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from car_racing.network import Actor#Var as Actor
from car_racing.network import Critic

from copg_optim.critic_functions import critic_update, get_advantage
import time

import json
import sys

import rcvip_simulator.VehicleModel as VehicleModel
from rcvip_env_function import getfreezeTimecollosionReachedrewardSingleAgent
import random

folder_location = 'trained_model/'
directory = os.path.join(folder_location, experiment_name, 'model')
if not os.path.exists(directory):
    os.makedirs(directory)

# save learning rate
with open(os.path.join(folder_location, experiment_name, 'param.json'),'w') as f:
        data = { 'critic_lr':critic_lr, 'actor_lr':actor_lr}
        data['critic_lr'] = critic_lr
        data['actor_lr'] = actor_lr
        data['num_episode'] = num_episode
        data['experiment_name'] = experiment_name
        data['batch_size'] = batch_size
        data['num_episode'] = num_episode
        json.dump(data,f,indent=4)

writer = SummaryWriter(os.path.join(folder_location, experiment_name, 'data'))

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu') # cpu is faster
print(f'using device: {device}')

n_action = 2
n_batch = 2
n_pred = 200
n_state = 6

vehicle_model = VehicleModel.VehicleModel(n_batch, device, track='rcp')

x0 = torch.zeros(n_batch, n_state)
u0 = torch.zeros(n_batch, n_action)

p1 = Actor(5,2, std=0.1)
q = Critic(5)

optim_p1 = torch.optim.SGD(p1.parameters(), actor_lr) # 3e-5,
optim_q = torch.optim.Adam(q.parameters(), critic_lr) # 8e-3

batch_size = 32
num_episode = 4000

try:
    last_checkpoint_eps = torch.load(os.path.join(folder_location,experiment_name,f'last_checkpoint_eps.pth'))
    print(f'resuming training from {last_checkpoint_eps}, optimizer state not loaded')
    t_eps = last_checkpoint_eps
    p1_state_dict = torch.load(os.path.join(folder_location,experiment_name,'model',f'agent1_{t_eps}.pth'))
    q_state_dict = torch.load(os.path.join(folder_location,experiment_name,'model',f'val_{t_eps}.pth'))
    p1.load_state_dict(p1_state_dict)
    q.load_state_dict(q_state_dict)
except FileNotFoundError:
    print('Starting new training')
    last_checkpoint_eps = 0

def simulate(device):
    curr_batch_size = batch_size

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

    # max episode length
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

    writer.add_scalar('Dist/variance_throttle_p1', dist.variance[0,0], t_eps)
    writer.add_scalar('Dist/variance_steer_p1', dist.variance[0,1], t_eps)
    writer.add_scalar('Reward/episode_reward_mean', episode_reward_mean, t_eps)
    writer.add_scalar('Reward/steps_reward_mean', step_reward_mean, t_eps)
    writer.add_scalar('Progress/mean_progress', episode_steps_mean, t_eps)

    return state_vec, action_vec, reward_vec, done_vec

def update(batch_mat_state1, batch_mat_action1, batch_mat_reward1, batch_mat_done, device):
    val1 = q(batch_mat_state1)
    val1 = val1.detach().to('cpu')
    next_value = 0  # because currently we end ony when its done which is equivalent to no next state
    returns_np1 = get_advantage(next_value, batch_mat_reward1, val1, batch_mat_done, gamma=0.99, tau=0.95)

    returns1 = torch.cat(returns_np1)
    advantage_mat1 = returns1.view(1,-1) - val1.transpose(0,1)

    state_gpu_p1 = batch_mat_state1
    returns1_gpu = returns1.view(-1, 1).to(device)

    for loss_critic, gradient_norm in critic_update(state_gpu_p1,returns1_gpu, q, optim_q):
        writer.add_scalar('Loss/critic', loss_critic, t_eps)
    ed_q_time = time.time()

    val1_p = advantage_mat1.to(device)
    writer.add_scalar('Advantage/agent1', advantage_mat1.mean(), t_eps)
    # calculate gradients
    dist_batch1 = p1(state_gpu_p1)
    log_probs1 = dist_batch1.log_prob(batch_mat_action1)

    objective = -log_probs1 * val1_p.transpose(0,1) # ob.backward will maximise this
    ob1 = objective.mean()
    optim_p1.zero_grad()
    ob1.backward()
    total_norm = 0
    total_norm = total_norm ** (1. / 2)
    optim_p1.step()
    writer.add_scalar('Objective/gradnorm1', total_norm, t_eps)

    writer.add_scalar('Entropy/agent1', dist_batch1.entropy().mean().detach(), t_eps)
    writer.add_scalar('Objective/gradp1', ob1.detach(), t_eps)
    writer.flush()

for t_eps in range(last_checkpoint_eps,num_episode):
    retval = simulate(device)
    batch_mat_state1, batch_mat_action1, batch_mat_reward1,  batch_mat_done = retval
    update(batch_mat_state1, batch_mat_action1, batch_mat_reward1, batch_mat_done, device)

    if t_eps%20==0:
        torch.save(p1.state_dict(),os.path.join(folder_location,experiment_name,'model',f'agent1_{t_eps}.pth'))
        torch.save(q.state_dict(),os.path.join(folder_location,experiment_name,'model',f'val_{t_eps}.pth'))
        torch.save(t_eps,os.path.join(folder_location,experiment_name,f'last_checkpoint_eps.pth'))
    sys.stdout.flush()

