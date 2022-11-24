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
        critic_lr = 1e-4 # default 0.008
        actor_lr = 1e-5 # default 3e-5
        experiment_name = sys.argv[1]
        original_batch_size = batch_size = 32
        num_episode = 15000
        print(f'Using hard-coded params')
else:
    print(f'usage1: python copg_rvip.py exp_name')
    print(f'usage2: python copg_rvip.py param_json_name')
    exit(1)

print(f'experiment_name: {experiment_name}')
print(f'critic_lr: {critic_lr}')
print(f'actor_lr:{actor_lr}')
print(f'batch_size: {batch_size}')
print(f'num_episode: {num_episode}')

import time
import random
import torch
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0,'..')
sys.path.insert(0,'../..')
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from car_racing.network import Actor
from car_racing.network import Critic
from copg_optim import RCoPG as CoPG
from copg_optim.critic_functions import critic_update, get_advantage


from common import *
import rcvip_simulator.VehicleModel as VehicleModel
import rcvip_simulator.Track as Track
from rcvip_env_function import getfreezeTimecollosionReachedreward
from util.timeUtil import execution_timer
t = execution_timer(True)


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

p1 = Actor(10,2, std=0.1).to(device)
p2 = Actor(10,2, std=0.1).to(device)
q = Critic(10).to(device)


try:
    last_checkpoint_eps = torch.load(os.path.join(folder_location,experiment_name,f'last_checkpoint_eps.pth'))
    print(f'resuming training from {last_checkpoint_eps}, optimizer state not loaded')
    t_eps = last_checkpoint_eps
    p1_state_dict = torch.load(os.path.join(folder_location,experiment_name,'model',f'agent1_{t_eps}.pth'))
    p2_state_dict = torch.load(os.path.join(folder_location,experiment_name,'model',f'agent2_{t_eps}.pth'))
    q_state_dict = torch.load(os.path.join(folder_location,experiment_name,'model',f'val_{t_eps}.pth'))
    p1.load_state_dict(p1_state_dict)
    p2.load_state_dict(p2_state_dict)
    q.load_state_dict(q_state_dict)
except FileNotFoundError:
    print('Starting new training')
    last_checkpoint_eps = 0

optim_q = torch.optim.Adam(q.parameters(), lr=critic_lr)
optim = CoPG(p1.parameters(),p2.parameters(), lr=actor_lr, device=device)

print(f'training for {num_episode} episodes')



# -------------- funs ----------

def simulate(device,t):
    t.s('prep')

    batch_size = original_batch_size
    mat_state1 =  torch.empty(0,batch_size,5,device = device)
    mat_state2 =  torch.empty(0,batch_size,5,device = device)
    mat_action1 = torch.empty(0,batch_size,2,device = device)
    mat_action2 = torch.empty(0,batch_size,2,device = device)
    mat_done =    torch.empty(0,batch_size,1,device = device)
    mat_reward1 = torch.empty(0,batch_size,1,device = device)
    mat_reward2 = torch.empty(0,batch_size,1,device = device)
    print(f'episode {t_eps}')

    avg_itr = 0

    #state = torch.zeros(n_batch, n_state)
    state_c1 = torch.zeros(batch_size, n_state).to(device)#state[:,0:6].view(6)
    state_c2 = torch.zeros(batch_size, n_state).to(device)#state[:, 6:12].view(6)
    init_p1 = torch.zeros((batch_size)).to(device) #5*torch.rand((batch_size))
    init_p2 = torch.zeros((batch_size)).to(device) #5*torch.rand((batch_size))
    state_c1[:,0] = init_p1
    state_c2[:,0] = init_p2
    # random initial state:  lateral_offset
    a = random.choice([-0.1,0.1])
    b = a*(-1)
    state_c1[:, 1] = a*torch.ones((batch_size))
    state_c2[:, 1] = b*torch.ones((batch_size))
    batch_mat_state1 =  torch.empty(0)
    batch_mat_state2 =  torch.empty(0)
    batch_mat_action1 = torch.empty(0)
    batch_mat_action2 = torch.empty(0)
    batch_mat_reward1 = torch.empty(0)
    batch_mat_done = torch.empty(0)

    itr = 0
    done = torch.tensor([False],device=device)
    done_c1 = torch.zeros((batch_size),device=device) <= -0.1
    done_c2 = torch.zeros((batch_size),device=device) <= -0.1
    prev_coll_c1 = torch.zeros((batch_size),device=device) <= -0.1
    prev_coll_c2 = torch.zeros((batch_size),device=device) <= -0.1
    counter1 = torch.zeros((batch_size),device=device)
    counter2 = torch.zeros((batch_size),device=device)
    t.e('prep')



    #for itr in range(50):
    while np.all(done.cpu().numpy()) == False:
        avg_itr+=1

        t.s('sample_policy')
        # drop the last state state_c1[:,5] is angular rate
        full_state1 = torch.cat([state_c1[:,0:5],state_c2[:,0:5]],dim=1)
        dist1 = p1(full_state1)
        action1 = dist1.sample()

        full_state2 = torch.cat([state_c2[:, 0:5], state_c1[:, 0:5]], dim=1)
        dist2 = p2(full_state2)
        action2 = dist2.sample()
        t.e('sample_policy')

        t.s('cat state')
        mat_state1 = torch.cat([mat_state1,   state_c1[:,0:5].view(-1,batch_size,5)],dim=0) # concate along dim = 0
        mat_state2 = torch.cat([mat_state2,   state_c2[:, 0:5].view(-1, batch_size, 5)], dim=0)
        mat_action1 = torch.cat([mat_action1, action1.view(-1, batch_size, 2)], dim=0)
        mat_action2 = torch.cat([mat_action2, action2.view(-1, batch_size, 2)], dim=0)
        t.e('cat state')


        prev_state_c1 = state_c1
        prev_state_c2 = state_c2


        t.s('dynamics')
        state_c1 = vehicle_model.dynModelBlendBatch(state_c1, action1,t)
        state_c2 = vehicle_model.dynModelBlendBatch(state_c2, action2,t)
        t.e('dynamics')


        t.s('reward')
        # if a sim is done, then freeze its state
        state_c1 = (state_c1.transpose(0, 1) * (~done_c1) + prev_state_c1.transpose(0, 1) * (done_c1)).transpose(0, 1)
        state_c2 = (state_c2.transpose(0, 1) * (~done_c2) + prev_state_c2.transpose(0, 1) * (done_c2)).transpose(0, 1)

        reward1, reward2, done_c1, done_c2, coll_c1, coll_c2, counter1, counter2 = getfreezeTimecollosionReachedreward(
                                                                     state_c1, state_c2,
                                                                     vehicle_model.getLocalBounds(state_c1[:, 0]),
                                                                     vehicle_model.getLocalBounds(state_c2[:, 0]),
                                                                     prev_state_c1, prev_state_c2, prev_coll_c1, prev_coll_c2, counter1, counter2,device=device)

        t.e('reward')

        
        # both done
        done = (done_c1) * (done_c2)
        # any done
        # done =  ~((~done_c1) * (~done_c2))


        mat_reward1 = torch.cat([mat_reward1.view(-1,batch_size,1),reward1.view(-1,batch_size,1)],dim=0)
        mat_done = torch.cat([mat_done.view(-1, batch_size, 1), ~done.view(-1, batch_size, 1)], dim=0)

        state_c1 = state_c1[~done]
        state_c2 = state_c2[~done]
        prev_coll_c1 = coll_c1[~done]#removing elements that died
        prev_coll_c2 = coll_c2[~done]#removing elements that died
        counter1 = counter1[~done]
        counter2 = counter2[~done]

        # TODO FIXME
        batch_size = state_c1.size(0)

        if batch_size<done.size(0):
            if batch_mat_action1.nelement() == 0:
                batch_mat_state1 = mat_state1.transpose(0, 1)[done].view(-1, 5)
                batch_mat_state2 = mat_state2.transpose(0, 1)[done].view(-1, 5)
                batch_mat_action1 = mat_action1.transpose(0, 1)[done].view(-1, 2)
                batch_mat_action2 = mat_action2.transpose(0, 1)[done].view(-1, 2)
                batch_mat_reward1 = mat_reward1.transpose(0, 1)[done].view(-1, 1)
                batch_mat_done = mat_done.transpose(0, 1)[done].view(-1, 1)
                # progress_done1 = batch_mat_state1[batch_mat_state1.size(0) - 1, 0] - batch_mat_state1[0, 0]
                # progress_done2 = batch_mat_state2[batch_mat_state2.size(0) - 1, 0] - batch_mat_state2[0, 0]
                progress_done1 = torch.sum(mat_state1.transpose(0, 1)[done][:,mat_state1.size(0)-1,0] - mat_state1.transpose(0, 1)[done][:,0,0])
                progress_done2 = torch.sum(mat_state2.transpose(0, 1)[done][:,mat_state2.size(0)-1,0] - mat_state2.transpose(0, 1)[done][:,0,0])
                element_deducted = ~(done_c1*done_c2)
                done_c1 = done_c1[element_deducted]
                done_c2 = done_c2[element_deducted]
            else:
                prev_size = batch_mat_state1.size(0)
                batch_mat_state1 = torch.cat([batch_mat_state1,mat_state1.transpose(0, 1)[done].view(-1,5)],dim=0)
                batch_mat_state2 = torch.cat([batch_mat_state2, mat_state2.transpose(0, 1)[done].view(-1, 5)],dim=0)
                batch_mat_action1 = torch.cat([batch_mat_action1, mat_action1.transpose(0, 1)[done].view(-1, 2)],dim=0)
                batch_mat_action2 = torch.cat([batch_mat_action2, mat_action2.transpose(0, 1)[done].view(-1, 2)],dim=0)
                batch_mat_reward1 = torch.cat([batch_mat_reward1, mat_reward1.transpose(0, 1)[done].view(-1, 1)],dim=0)
                batch_mat_done = torch.cat([batch_mat_done, mat_done.transpose(0, 1)[done].view(-1, 1)],dim=0)
                progress_done1 = progress_done1 + torch.sum(mat_state1.transpose(0, 1)[done][:, mat_state1.size(0) - 1, 0] -
                                           mat_state1.transpose(0, 1)[done][:, 0, 0])
                progress_done2 = progress_done2 + torch.sum(mat_state2.transpose(0, 1)[done][:, mat_state2.size(0) - 1, 0] -
                                           mat_state2.transpose(0, 1)[done][:, 0, 0])
                # progress_done1 = progress_done1 + batch_mat_state1[batch_mat_state1.size(0) - 1, 0] - batch_mat_state1[prev_size, 0]
                # progress_done2 = progress_done2 + batch_mat_state2[batch_mat_state2.size(0) - 1, 0] - batch_mat_state2[prev_size, 0]
                element_deducted = ~(done_c1*done_c2)
                done_c1 = done_c1[element_deducted]
                done_c2 = done_c2[element_deducted]

            mat_state1 = mat_state1.transpose(0, 1)[~done].transpose(0, 1)
            mat_state2 = mat_state2.transpose(0, 1)[~done].transpose(0, 1)
            mat_action1 = mat_action1.transpose(0, 1)[~done].transpose(0, 1)
            mat_action2 = mat_action2.transpose(0, 1)[~done].transpose(0, 1)
            mat_reward1 = mat_reward1.transpose(0, 1)[~done].transpose(0, 1)
            mat_done = mat_done.transpose(0, 1)[~done].transpose(0, 1)

        # print(avg_itr,~done.size(0))

        # writer.add_scalar('Reward/agent1', reward1, t_eps)
        itr = itr + 1

        if np.all(done.cpu().numpy()) == True or batch_mat_state1.size(0)>900 or itr>400:# or itr>900: #brak only if all elements in the array are true
            prev_size = batch_mat_state1.size(0)
            batch_mat_state1 = torch.cat([batch_mat_state1, mat_state1.transpose(0, 1).reshape(-1, 5)],dim=0)
            batch_mat_state2 = torch.cat([batch_mat_state2, mat_state2.transpose(0, 1).reshape(-1, 5)],dim=0)
            batch_mat_action1 = torch.cat([batch_mat_action1, mat_action1.transpose(0, 1).reshape(-1, 2)],dim=0)
            batch_mat_action2 = torch.cat([batch_mat_action2, mat_action2.transpose(0, 1).reshape(-1, 2)],dim=0)
            batch_mat_reward1 = torch.cat([batch_mat_reward1, mat_reward1.transpose(0, 1).reshape(-1, 1)],dim=0) #should i create a false or true array?
            print(f"all episodes done at step {itr}")
            mat_done[mat_done.size(0)-1,:,:] = torch.ones((mat_done[mat_done.size(0)-1,:,:].shape))>=2 # creating a true array of that shape
            #print(mat_done.shape, batch_mat_done.shape)
            if batch_mat_done.nelement() == 0:
                batch_mat_done = mat_done.transpose(0, 1).reshape(-1, 1)
                progress_done1 = 0
                progress_done2 = 0
            else:
                batch_mat_done = torch.cat([batch_mat_done, mat_done.transpose(0, 1).reshape(-1, 1)], dim=0)
            if prev_size == batch_mat_state1.size(0):
                progress_done1 = progress_done1
                progress_done2 = progress_done2
            else:
                progress_done1 = progress_done1 + torch.sum(mat_state1.transpose(0, 1)[:, mat_state1.size(0) - 1, 0] -
                                           mat_state1.transpose(0, 1)[:, 0, 0])
                progress_done2 = progress_done2 + torch.sum(mat_state2.transpose(0, 1)[:, mat_state2.size(0) - 1, 0] -
                                           mat_state2.transpose(0, 1)[:, 0, 0])
            #print(batch_mat_done.shape)
            # print("done", itr)
            break

    # print(avg_itr)

    dist = {'variance_throttle_p1':dist1.variance[0,0],
            'variance_steer_p1': dist1.variance[0,1],
            'variance_throttle_p2': dist2.variance[0,0],
            'variance_steer_p2': dist2.variance[0,1]}
    writer.add_scalars('control_var', dist, t_eps)


    writer.add_scalar('Reward/mean', batch_mat_reward1.mean(), t_eps)
    writer.add_scalar('Reward/sum', batch_mat_reward1.sum(), t_eps)

    print(f'batch size {batch_size}')
    print(f'progress_done1 {progress_done1}')
    print(f'final_p1 {progress_done1/batch_size}')
    writer.add_scalar('Progress/final_p1', progress_done1/batch_size, t_eps)
    writer.add_scalar('Progress/final_p2', progress_done2/batch_size, t_eps)
    writer.add_scalar('Progress/trajectory_length', itr, t_eps)
    writer.add_scalar('Progress/agent1', batch_mat_state1[:,0].mean(), t_eps)
    writer.add_scalar('Progress/agent2', batch_mat_state2[:,0].mean(), t_eps)
    return batch_mat_state1, batch_mat_action1, batch_mat_reward1, batch_mat_state2, batch_mat_action2, batch_mat_done

def update(batch_mat_state1, batch_mat_action1, batch_mat_reward1, batch_mat_state2, batch_mat_action2, batch_mat_done, device,t):
    val1 = q(torch.cat([batch_mat_state1,batch_mat_state2],dim=1).to(device))
    #NOTE should this be detached?
    #val1 = val1.detach().to('cpu')
    val1 = val1.detach()
    next_value = 0  # because currently we end ony when its done which is equivalent to no next state
    returns_np1 = get_advantage(next_value, batch_mat_reward1, val1, batch_mat_done, gamma=0.99, tau=0.95,device=device)

    returns1 = torch.cat(returns_np1)
    advantage_mat1 = returns1.view(1,-1) - val1.transpose(0,1)

    # val2 = q(torch.stack(mat_state2))
    # val2 = val2.detach()
    # next_value = 0  # because currently we end ony when its done which is equivalent to no next state
    # returns_np2 = get_advantage(next_value, torch.stack(mat_reward2), val2, torch.stack(mat_done), gamma=0.99, tau=0.95)
    #
    # returns2 = torch.cat(returns_np2)
    # advantage_mat2 = returns2 - val2.transpose(0,1)
    state_gpu_p1 = torch.cat([batch_mat_state1, batch_mat_state2], dim=1).to(device)
    state_gpu_p2 = torch.cat([batch_mat_state2, batch_mat_state1], dim=1).to(device)
    returns1_gpu = returns1.view(-1, 1).to(device)

    for loss_critic, gradient_norm in critic_update(state_gpu_p1,returns1_gpu, q, optim_q):
        writer.add_scalar('Loss/critic', loss_critic, t_eps)
        #print('critic_update')
    ed_q_time = time.time()
    # print('q_time',ed_q_time-st_q_time)

    #val1_p = -advantage_mat1#val1.detach()
    val1_p = advantage_mat1.to(device)
    writer.add_scalar('Advantage/agent1', advantage_mat1.mean(), t_eps)
    # st_time = time.time()
    # calculate gradients
    batch_mat_action1_gpu = batch_mat_action1.to(device)
    dist_batch1 = p1(state_gpu_p1)
    log_probs1_inid = dist_batch1.log_prob(batch_mat_action1_gpu)
    log_probs1 = log_probs1_inid.sum(1)

    batch_mat_action2_gpu = batch_mat_action2.to(device)
    dist_batch2 = p2(state_gpu_p2)
    log_probs2_inid = dist_batch2.log_prob(batch_mat_action2_gpu)
    log_probs2 = log_probs2_inid.sum(1)

    objective = log_probs1*log_probs2*(val1_p)
    if objective.size(0)!=1:
        raise 'error'

    ob = objective.mean()

    s_log_probs1 = log_probs1[0:log_probs1.size(0)].clone() # otherwise it doesn't change values
    s_log_probs2 = log_probs2[0:log_probs2.size(0)].clone()

    #if horizon is 4, log_probs1.size =4 and for loop will go till [0,3]

    mask = batch_mat_done.to(device)

    s_log_probs1[0] = 0
    s_log_probs2[0] = 0

    for i in range(1,log_probs1.size(0)):
        s_log_probs1[i] = torch.add(s_log_probs1[i - 1], log_probs1[i-1])*mask[i-1]
        s_log_probs2[i] = torch.add(s_log_probs2[i - 1], log_probs2[i-1])*mask[i-1]

    objective2 = s_log_probs1[1:s_log_probs1.size(0)]*log_probs2[1:log_probs2.size(0)]*(val1_p[0,1:val1_p.size(1)])
    ob2 = objective2.sum()/(objective2.size(0)-batch_size+1)

    objective3 = log_probs1[1:log_probs1.size(0)]*s_log_probs2[1:s_log_probs2.size(0)]*(val1_p[0,1:val1_p.size(1)])
    ob3 = objective3.sum()/(objective2.size(0)-batch_size+1)
    #print(advantage_mat1.mean(),advantage_mat2.mean())
    lp1 = log_probs1*val1_p
    lp1=lp1.mean()
    lp2 = log_probs2*val1_p
    lp2=lp2.mean()
    optim.zero_grad()
    optim.step(ob, ob+ob2+ob3, lp1,lp2)

    # torch.autograd.grad(ob2.mean(), list(p1.parameters), create_graph=True, retain_graph=True)
    ed_time = time.time()

    writer.add_scalar('Entropy/agent1', dist_batch1.entropy().mean().detach(), t_eps)
    writer.add_scalar('Entropy/agent2', dist_batch2.entropy().mean().detach(), t_eps)
    writer.add_scalar('Objective/gradfg_t1', ob.detach(), t_eps)
    writer.add_scalar('Objective/gradfg_t2', ob2.detach(), t_eps)
    writer.add_scalar('Objective/gradfg_t3', ob3.detach(), t_eps)
    writer.add_scalar('Objective/gradf', lp1.detach(), t_eps)
    writer.add_scalar('Objective/gradg', lp2.detach(), t_eps)

    norm_gx, norm_gy, norm_px, norm_py, norm_cgx, norm_cgy, timer, itr_num, norm_cgx_cal, norm_cgy_cal, norm_vx, norm_vy, norm_mx, norm_my = optim.getinfo()
    writer.add_scalar('grad/norm_gx', norm_gx, t_eps)
    writer.add_scalar('grad/norm_gy', norm_gy, t_eps)
    writer.add_scalar('grad/norm_px', norm_px, t_eps)
    writer.add_scalar('grad/norm_py', norm_py, t_eps)
    writer.add_scalar('inverse/itr_num', itr_num, t_eps)
    writer.add_scalar('inverse/timer', timer, t_eps)
    writer.add_scalar('grad/norm_vx', norm_vx, t_eps)
    writer.add_scalar('grad/norm_vy', norm_vy, t_eps)
    writer.add_scalar('grad/norm_mx', norm_mx, t_eps)
    writer.add_scalar('grad/norm_my', norm_my, t_eps)
    writer.add_scalar('grad/norm_cgx', norm_cgx, t_eps)
    writer.add_scalar('grad/norm_cgy', norm_cgy, t_eps)
    writer.add_scalar('grad/norm_cgx_cal', norm_cgx_cal, t_eps)
    writer.add_scalar('grad/norm_cgy_cal', norm_cgy_cal, t_eps)
    writer.flush()

# TODO, test performance against benchmark
def test():
    pass

# simulate episodes
for t_eps in range(last_checkpoint_eps,num_episode):
    t.s() # full cpu on laptop: 0.27Hz, sim and prep all take ~50%

    t.s('sim')
    retval = simulate(device,t)
    batch_mat_state1, batch_mat_action1, batch_mat_reward1, batch_mat_state2, batch_mat_action2, batch_mat_done = retval
    t.e('sim')

    t.s('update')
    update(batch_mat_state1, batch_mat_action1, batch_mat_reward1, batch_mat_state2, batch_mat_action2, batch_mat_done, device,t)
    t.e('update')

    t.s('test')
    test()
    t.e('test')

    if t_eps%20==0:
        torch.save(p1.state_dict(),os.path.join(folder_location,experiment_name,'model',f'agent1_{t_eps}.pth'))
        torch.save(p2.state_dict(),os.path.join(folder_location,experiment_name,'model',f'agent2_{t_eps}.pth'))
        torch.save(q.state_dict(),os.path.join(folder_location,experiment_name,'model',f'val_{t_eps}.pth'))
        torch.save(t_eps,os.path.join(folder_location,experiment_name,f'last_checkpoint_eps.pth'))
    t.e()
    sys.stdout.flush()
t.summary()
