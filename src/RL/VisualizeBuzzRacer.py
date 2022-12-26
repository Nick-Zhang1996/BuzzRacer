import sys
sys.path.insert(0,'..')
import os
import gym
import torch
from gym_envs.BuzzRacer.BuzzRacerEnv import BuzzRacerEnv
from gym.wrappers import TimeLimit

import numpy as np
from DQN import DQN


env = TimeLimit(BuzzRacerEnv(render_mode='human'),max_episode_steps=1000)
n_observations = env.observation_space.shape[0]
n_actions = env.action_space.shape[0]
observation, info = env.reset(seed=42)
episode_len_vec = []
episode_reward_vec = []

'''
device = 'cpu'
policy = DQN(n_observations, n_actions).to(device)
directory = os.path.join('models',  'cartpole_dqn')
policy_state_dict = torch.load(os.path.join(directory,'target_net_480.pth'))
torch.no_grad()
policy.load_state_dict(policy_state_dict)
'''

def DQN_policy(observation):
    observation = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
    action = policy(observation).max(1)[1].view(1, 1)
    return action.item()

def random_policy(observation):
    global env
    action = env.action_space.sample()
    return action

policy = random_policy

current_episode_len = 0
current_episode_reward = 0
for _ in range(1000):
    action = policy(observation)
    observation, reward, terminated, truncated, info = env.step(action)
    current_episode_len += 1
    current_episode_reward += reward

    if terminated or truncated:
        print(f'reset at {current_episode_len}, reward {current_episode_reward}')
        episode_len_vec.append(current_episode_len)
        episode_reward_vec.append(current_episode_reward)
        current_episode_len = 0
        current_episode_reward = 0
        observation, info = env.reset()
env.close()
print(f'mean episode len = {np.mean(episode_len_vec)}')
print(f'mean episode reward = {np.mean(episode_len_vec)}')
