# visualize cart pole
import os
import gym
import numpy as np
from DQN import DQN
import torch

env = gym.make("CartPole-v1", render_mode="human")
n_observations = env.observation_space.shape[0]
n_actions = env.action_space.n
observation, info = env.reset(seed=42)
episode_len_vec = []


#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'
policy = DQN(n_observations, n_actions).to(device)
directory = os.path.join('models',  'cartpole_dqn')
policy_state_dict = torch.load(os.path.join(directory,'target_net_480.pth'))
torch.no_grad()
policy.load_state_dict(policy_state_dict)


current_episode_len = 0
for _ in range(1000):
    #action = policy(observation)  # User-defined policy function
    #action = env.action_space.sample() # mean episode len 22
    observation = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
    action = policy(observation).max(1)[1].view(1, 1)
    observation, reward, terminated, truncated, info = env.step(action.item())
    current_episode_len += 1

    if terminated or truncated:
        print('reset')
        episode_len_vec.append(current_episode_len)
        current_episode_len = 0
        observation, info = env.reset()
env.close()
print(f'mean episode len = {np.mean(episode_len_vec)}')
