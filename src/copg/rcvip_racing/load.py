import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0,'..')
from copg_optim import RCoPG as CoPG
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from car_racing.network import Actor
from car_racing.network import Critic

from copg_optim.critic_functions import critic_update, get_advantage
import time
import random

import json
import sys

import rcvip_simulator.VehicleModel as VehicleModel
import rcvip_simulator.Track as Track
from rcvip_env_function import getfreezeTimecollosionReachedreward

folder_location = 'trained_model/'
experiment_name = 'rcvip_half_lr/copg/'
directory = './' + folder_location + experiment_name + 'model'

device = torch.device('cpu') # cpu is faster
p1 = Actor(10,2, std=0.1).to(device)
p2 = Actor(10,2, std=0.1).to(device)
q = Critic(10).to(device)

t_eps = 80
p1_state_dict = torch.load('./' + folder_location + experiment_name + 'model/agent1_' + str(t_eps) + ".pth")
p2_state_dict = torch.load('./' + folder_location + experiment_name + 'model/agent2_' + str(t_eps) + ".pth")
q_state_dict = torch.load('./' + folder_location + experiment_name + 'model/val_' + str(t_eps) + ".pth")

p1.load_state_dict(p1_state_dict)
p2.load_state_dict(p2_state_dict)
q.load_state_dict(q_state_dict)
