import sys
sys.path.insert(0,'..')
import gym
from gym_envs.BuzzRacer.BuzzRacerEnv import BuzzRacerEnv
from gym.wrappers import TimeLimit

env = TimeLimit(BuzzRacerEnv(render_mode=None),max_episode_steps=1000)

observation, info = env.reset(seed=42)
for i in range(1000):
    #action = policy(observation)  # User-defined policy function
    #action = env.action_space.sample()
    action = (1.0,0.1)
    observation, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        print(f'reset at {i}')
        observation, info = env.reset()
env.close()
