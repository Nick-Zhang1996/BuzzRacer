# compare two models
# generate a simulation log to be replayed later
import torch
import sys
import os
# sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0,'..') # inorder to run within the folder
import numpy as np
import json
import pickle



# rcp or orca
track = 'orca'

# player name, doesn't mean anything
player1 = 'CoPG'
player2 = 'CoPG'
# number of experiments to run
init_size  = 10000
# time steps for an experiment, dt = 0.03
duration = 3000

# models to use
# pretrained model: CoPG,GDA, TRCoPO, TRGDA 
# pretrained: 'pretrained_models/CoPG.pth'
i = 9980
p1_policy_pth = f"trained_model/rcvip_half_lr/copg/model/agent1_{i}.pth"
p2_policy_pth = f"trained_model/rcvip_half_lr/copg/model/agent1_{i}.pth"

#i = 2160
#p1_policy_pth = f"../car_racing/trained_model/original/copg/model/agent1_{i}.pth"
#p2_policy_pth = f"../car_racing/trained_model/original/copg/model/agent1_{i}.pth"

# -----------------------


print(f'running on {track} track')
print(f'running {init_size} experiments')
print(f'player 1 policy {p1_policy_pth}')
print(f'player 2 policy {p2_policy_pth}')
if track=='rcp':
    from network import Actor as Actor
    from rcvip_env_function import getNFcollosionreward
    import rcvip_simulator.VehicleModel as VehicleModel
    import rcvip_simulator.Track as Track
elif track=='orca':
    from car_racing.network import Actor as Actor
    from car_racing.orca_env_function import getNFcollosionreward
    import car_racing_simulator.VehicleModel as VehicleModel
    import car_racing_simulator.Track as Track


p1 = Actor(10, 2, std=0.1)
p2 = Actor(10, 2, std=0.1)

p1.load_state_dict(torch.load(p1_policy_pth))
p2.load_state_dict(torch.load(p2_policy_pth))

config = json.load(open('config.json'))

device = torch.device("cpu")

vehicle_model = VehicleModel.VehicleModel(config["n_batch"], device, config,track=track)

mat_action1 = []
mat_action2 = []

mat_state1 = []
mat_reward1 = []
mat_done = []

mat_state2 = []
global_coordinates1 = []
curvilinear_coordinates1 = []

global_coordinates2 = []
curvilinear_coordinates2 = []
curr_batch_size = init_size

state_c1_vec = []
state_c2_vec = []
action_c1_vec = []
action_c2_vec = []


state_c1 = torch.zeros(curr_batch_size, config["n_state"])  # state[:, 6:12].view(6)
state_c2 = torch.zeros(curr_batch_size, config["n_state"])  # state[:, 6:12].view(6)
state_c1[:, 0] = torch.zeros((curr_batch_size))#torch.rand((curr_batch_size))
state_c2[:, 0] = torch.zeros((curr_batch_size))#torch.FloatTensor([2.0])#torch.rand((curr_batch_size))

a = torch.rand(curr_batch_size)
a_linear = (a>0.5)*torch.ones(curr_batch_size)
state_c1[:, 1] = a_linear*0.2 - 0.1
state_c2[:, 1] = -state_c1[:, 1]
# state_c1[:, 1] = -0.1#torch.zeros((curr_batch_size))#torch.rand((curr_batch_size))
# state_c2[:, 1] = 0.1#torch.zeros((curr_batch_size))#torch.FloatTensor([2.0])#torch.rand((curr_batch_size))
done_c1 = torch.zeros((curr_batch_size)) <= -0.1
done_c2 = torch.zeros((curr_batch_size)) <= -0.1
prev_coll_c1 = torch.zeros((curr_batch_size)) <= -0.1
prev_coll_c2 = torch.zeros((curr_batch_size)) <= -0.1
counter1 = torch.zeros((curr_batch_size))
counter2 = torch.zeros((curr_batch_size))
over_mat = []
overtakings = torch.zeros((curr_batch_size))
prev_leading_player = torch.cat([torch.zeros(int(curr_batch_size/2)) <= 0.1,torch.zeros(int(curr_batch_size/2)) <= -0.1])
c1_out=0
c2_out=0
t=0
a_win=0
b_win=0
overtakings_p1 = 0
overtakings_p2 = 0
for i in range(duration):

    #sample action from random policy
    dist1 = p1(torch.cat([state_c1[:, 0:5], state_c2[:, 0:5]], dim=1))
    action1 = dist1.sample()

    dist2 = p2(torch.cat([state_c2[:, 0:5], state_c1[:, 0:5]], dim=1))
    action2 = dist2.sample()

    mat_state1.append(state_c1[0:5])
    mat_action1.append(action1.detach())

    prev_state_c1 = state_c1
    prev_state_c2 = state_c2

    # log state
    state_c1_vec.append(state_c1[0].detach().numpy())
    state_c2_vec.append(state_c2[0].detach().numpy())
    action_c1_vec.append(action1[0].detach().numpy())
    action_c2_vec.append(action2[0].detach().numpy())

    # advance state
    state_c1 = vehicle_model.dynModelBlendBatch(state_c1.view(-1, 6), action1.view(-1, 2)).view(-1, 6)
    state_c2 = vehicle_model.dynModelBlendBatch(state_c2.view(-1, 6), action2.view(-1, 2)).view(-1, 6)

    # if it's done, hold last state
    state_c1 = (state_c1.transpose(0, 1) * (~done_c1) + prev_state_c1.transpose(0, 1) * (done_c1)).transpose(0, 1)
    state_c2 = (state_c2.transpose(0, 1) * (~done_c2) + prev_state_c2.transpose(0, 1) * (done_c2)).transpose(0, 1)

    # evaluate current step
    reward1, reward2, done_c1, done_c2,state_c1, state_c2, n_c1, n_c2  = getNFcollosionreward(state_c1, state_c2,
                                                                      vehicle_model.getLocalBounds(state_c1[:, 0]),
                                                                      vehicle_model.getLocalBounds(state_c2[:, 0]),
                                                                      prev_state_c1, prev_state_c2)

    done = ((done_c1) * (done_c2))
    remaining_xo = ~done
    # prev_coll_c1 = coll_c1[remaining_xo]  # removing elements that died
    # prev_coll_c2 = coll_c2[remaining_xo]
    counter1 = counter1[remaining_xo]
    counter2 = counter2[remaining_xo]
    # check for collision
    c1_out = c1_out + n_c1
    c2_out = c2_out + n_c2

    #check for overtake state_c1[:,2]
    leading_player = torch.ones(state_c1.size(0))*((state_c1[:,0]-state_c2[:,0])>0)# True means 1 is leading false means other is leading
    overtakings = overtakings + torch.ones(leading_player.size(0))*(leading_player!=prev_leading_player)

    overtakings_p1_bool= (leading_player!=prev_leading_player)*(leading_player==1)
    overtakings_p1 = overtakings_p1  + torch.sum(torch.ones(leading_player.size(0))*overtakings_p1_bool)
    overtakings_p2_bool= (leading_player!=prev_leading_player)*(leading_player==0)
    overtakings_p2 = overtakings_p2  + torch.sum(torch.ones(leading_player.size(0))*overtakings_p2_bool)
    prev_leading_player = leading_player[remaining_xo]
    out_state_c1 = state_c1[~remaining_xo]
    out_state_c2 = state_c2[~remaining_xo]
    state_c1 = state_c1[remaining_xo]
    state_c2 = state_c2[remaining_xo]
    curr_batch_size = state_c1.size(0)

    if curr_batch_size < remaining_xo.size(0):
        t=t+1
        # if t==1:
        #     print(i)
        a_win = a_win + torch.sum(torch.ones(out_state_c1.size(0))*(out_state_c1[:,0]>out_state_c2[:,0]))
        b_win = b_win + torch.sum(torch.ones(out_state_c1.size(0))*(out_state_c1[:,0]<out_state_c2[:,0]))
        over_mat.append(torch.sum(overtakings[~remaining_xo]))
        element_deducted = ~(done_c1 * done_c2)
        done_c1 = done_c1[element_deducted]
        done_c2 = done_c2[element_deducted]
        overtakings = overtakings[remaining_xo]
        # print(over_mat)

    if np.all(done.numpy()) == True or i==1999:
    # if ((done_c1) * (done_c2)):
    #     print(torch.sum(torch.stack(over_mat)), c1_out,c2_out, a_win,b_win, overtakings_p1,overtakings_p2)
        print('Normalized score for races between ' + player1 + ' vs ' + player2,)
        print('Overtakes per lap', np.array(torch.sum(torch.stack(over_mat))/init_size))
        print( player1 + ' Won:', np.array(a_win / init_size))
        print( player2 + ' Won:', np.array(b_win / init_size))
        print( player1 + ' Collisions:', np.array(c1_out/init_size))
        print( player2 + ' Collisions:', np.array(c2_out/init_size))
        print( player1 + ' Overtake:', np.array(overtakings_p1/init_size))
        print( player2 + ' Overtake:', np.array(overtakings_p2/init_size))

        # print("done", i)
        break

# save state and action history

state_c1_vec = np.array(state_c1_vec)
state_c2_vec = np.array(state_c2_vec)
action_c1_vec = np.array(action_c1_vec)
action_c2_vec = np.array(action_c2_vec)

c1 = np.hstack([state_c1_vec, action_c1_vec])[:,np.newaxis,:]
c2 = np.hstack([state_c2_vec, action_c2_vec])[:,np.newaxis,:]
data = np.hstack([c1,c2])
print(data.shape)

with open('logs/log.p','wb') as f:
    pickle.dump(data,f)
