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

n_control = 2
n_batch = 2
n_pred = 200
n_state = 6

vehicle_model = VehicleModel.VehicleModel(n_batch, device, track='rcp')

x0 = torch.zeros(n_batch, n_state)
u0 = torch.zeros(n_batch, n_control)

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
    mat_action1 = []

    mat_state1 = []
    mat_reward1 = []
    mat_done = []

    avg_itr = 0

    curr_batch_size = batch_size

    state_c1 = torch.zeros(curr_batch_size, n_state)#state[:,0:6].view(6)
    init_p1 = torch.zeros((curr_batch_size)) #5*torch.rand((curr_batch_size))
    state_c1[:,0] = init_p1
    a = random.choice([-0.1,0.1])
    b = a*(-1)
    state_c1[:, 1] = a*torch.ones((curr_batch_size))
    batch_mat_state1 = torch.empty(0)
    batch_mat_action1 = torch.empty(0)
    batch_mat_reward1 = torch.empty(0)
    batch_mat_done = torch.empty(0)

    itr = 0
    done = torch.tensor([False])
    done_c1 = torch.zeros((curr_batch_size)) <= -0.1

    while np.all(done.numpy()) == False:
        avg_itr+=1

        dist1 = p1(state_c1[:,0:5])
        action1 = dist1.sample().to('cpu')


        if itr>0:
            mat_state1 = torch.cat([mat_state1.view(-1,curr_batch_size,5),state_c1[:,0:5].view(-1,curr_batch_size,5)],dim=0) # concate along dim = 0
            mat_action1 = torch.cat([mat_action1.view(-1, curr_batch_size, 2), action1.view(-1, curr_batch_size, 2)], dim=0)
        else:
            mat_state1 = state_c1[:,0:5]
            mat_action1 = action1

        prev_state_c1 = state_c1

        state_c1 = vehicle_model.dynModelBlendBatch(state_c1.view(-1,6), action1.view(-1,2)).view(-1,6)

        state_c1 = (state_c1.transpose(0, 1) * (~done_c1) + prev_state_c1.transpose(0, 1) * (done_c1)).transpose(0, 1)

        '''
        reward1, reward2, done_c1, done_c2, coll_c1, coll_c2, counter1, counter2 = getfreezeTimecollosionReachedreward(state_c1, state_c2,
                                                                     vehicle_model.getLocalBounds(state_c1[:, 0]),
                                                                     vehicle_model.getLocalBounds(state_c2[:, 0]),
                                                                     prev_state_c1, prev_state_c2, prev_coll_c1, prev_coll_c2, counter1, counter2,device)
        '''
        bounds = vehicle_model.getLocalBounds(state_c1[:, 0])
        reward1,  done_c1 = getfreezeTimecollosionReachedrewardSingleAgent(state_c1, bounds ,prev_state_c1,  device)
        done = done_c1
        mask_ele = ~done

        if itr>0:
            mat_reward1 = torch.cat([mat_reward1.view(-1,curr_batch_size,1),reward1.view(-1,curr_batch_size,1)],dim=0) # concate along dim = 0
            mat_done = torch.cat([mat_done.view(-1, curr_batch_size, 1), mask_ele.view(-1, curr_batch_size, 1)], dim=0)
        else:
            mat_reward1 = reward1
            mat_done = mask_ele

        remaining_xo = ~done

        state_c1 = state_c1[remaining_xo]

        curr_batch_size = state_c1.size(0)

        if curr_batch_size<remaining_xo.size(0):
            if batch_mat_action1.nelement() == 0:
                batch_mat_state1 = mat_state1.transpose(0, 1)[~remaining_xo].view(-1, 5)
                batch_mat_action1 = mat_action1.transpose(0, 1)[~remaining_xo].view(-1, 2)
                batch_mat_reward1 = mat_reward1.transpose(0, 1)[~remaining_xo].view(-1, 1)
                batch_mat_done = mat_done.transpose(0, 1)[~remaining_xo].view(-1, 1)
                progress_done1 = torch.sum(mat_state1.transpose(0, 1)[~remaining_xo][:,mat_state1.size(0)-1,0] - mat_state1.transpose(0, 1)[~remaining_xo][:,0,0])
                element_deducted = ~done_c1
                done_c1 = done_c1[element_deducted]
            else:
                prev_size = batch_mat_state1.size(0)
                batch_mat_state1 = torch.cat([batch_mat_state1,mat_state1.transpose(0, 1)[~remaining_xo].view(-1,5)],dim=0)
                batch_mat_action1 = torch.cat([batch_mat_action1, mat_action1.transpose(0, 1)[~remaining_xo].view(-1, 2)],dim=0)
                batch_mat_reward1 = torch.cat([batch_mat_reward1, mat_reward1.transpose(0, 1)[~remaining_xo].view(-1, 1)],dim=0)
                batch_mat_done = torch.cat([batch_mat_done, mat_done.transpose(0, 1)[~remaining_xo].view(-1, 1)],dim=0)
                progress_done1 = progress_done1 + torch.sum(mat_state1.transpose(0, 1)[~remaining_xo][:, mat_state1.size(0) - 1, 0] -
                                           mat_state1.transpose(0, 1)[~remaining_xo][:, 0, 0])
                element_deducted = ~done_c1
                done_c1 = done_c1[element_deducted]

            mat_state1 = mat_state1.transpose(0, 1)[remaining_xo].transpose(0, 1)
            mat_action1 = mat_action1.transpose(0, 1)[remaining_xo].transpose(0, 1)
            mat_reward1 = mat_reward1.transpose(0, 1)[remaining_xo].transpose(0, 1)
            mat_done = mat_done.transpose(0, 1)[remaining_xo].transpose(0, 1)

        itr = itr + 1

        if np.all(done.numpy()) == True or batch_mat_state1.size(0)>900 or itr>400:# or itr>900: #brak only if all elements in the array are true
            prev_size = batch_mat_state1.size(0)
            batch_mat_state1 = torch.cat([batch_mat_state1, mat_state1.transpose(0, 1).reshape(-1, 5)],dim=0)
            batch_mat_action1 = torch.cat([batch_mat_action1, mat_action1.transpose(0, 1).reshape(-1, 2)],dim=0)
            batch_mat_reward1 = torch.cat([batch_mat_reward1, mat_reward1.transpose(0, 1).reshape(-1, 1)],dim=0) #should i create a false or true array?

            print(f"episode: {t_eps} all episodes done at step {itr}")
            mat_done[mat_done.size(0)-1,:,:] = torch.ones((mat_done[mat_done.size(0)-1,:,:].shape))>=2 # creating a true array of that shape
            if batch_mat_done.nelement() == 0:
                batch_mat_done = mat_done.transpose(0, 1).reshape(-1, 1)
                progress_done1 = 0
            else:
                batch_mat_done = torch.cat([batch_mat_done, mat_done.transpose(0, 1).reshape(-1, 1)], dim=0)
            if prev_size == batch_mat_state1.size(0):
                progress_done1 = progress_done1
            else:
                progress_done1 = progress_done1 + torch.sum(mat_state1.transpose(0, 1)[:, mat_state1.size(0) - 1, 0] -
                                           mat_state1.transpose(0, 1)[:, 0, 0])
            break

    # print(avg_itr)

    writer.add_scalar('Dist/variance_throttle_p1', dist1.variance[0,0], t_eps)
    writer.add_scalar('Dist/variance_steer_p1', dist1.variance[0,1], t_eps)
    writer.add_scalar('Reward/mean', batch_mat_reward1.mean(), t_eps)
    writer.add_scalar('Reward/sum', batch_mat_reward1.sum(), t_eps)
    writer.add_scalar('Progress/final_p1', progress_done1/batch_size, t_eps)
    writer.add_scalar('Progress/trajectory_length', itr, t_eps)
    writer.add_scalar('Progress/agent1', batch_mat_state1[:,0].mean(), t_eps)
    breakpoint()
    return batch_mat_state1, batch_mat_action1, batch_mat_reward1,  batch_mat_done

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

