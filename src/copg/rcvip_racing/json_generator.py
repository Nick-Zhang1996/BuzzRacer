import os
import json
from itertools import product

folder_location = 'trained_model/'

critic_lr_vec = [1e-3,1e-4,1e-5]
actor_lr_vec = [1e-3,1e-4,1e-5]

for i,comb in enumerate(product(critic_lr_vec,actor_lr_vec)):
    critic_lr = comb[0]
    actor_lr = comb[1]
    experiment_name = f'exp{i}'
    batch_size = 8
    num_episode = 10000

    with open(os.path.join('exp_configs',f'{experiment_name}.json'),'w') as f:
            data = { 'critic_lr':critic_lr, 'actor_lr':actor_lr}
            data['critic_lr'] = critic_lr
            data['actor_lr'] = actor_lr
            data['num_episode'] = num_episode
            data['experiment_name'] = experiment_name
            data['batch_size'] = batch_size
            data['num_episode'] = num_episode
            json.dump(data,f,indent=4)
