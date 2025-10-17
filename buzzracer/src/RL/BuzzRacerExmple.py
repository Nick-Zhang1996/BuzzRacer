import sys
sys.path.insert(0,'..')
import gymnasium as gym
from gym_envs.BuzzRacer.BuzzRacerEnv import BuzzRacerEnv
from gymnasium.wrappers import TimeLimit

env = TimeLimit(BuzzRacerEnv(render_mode='human'),max_episode_steps=1000)

observation, info = env.reset(seed=42)
current_episode_len = 0
steps = 0

for _ in range(1000):
    #action = policy(observation)  # User-defined policy function
    #action = env.action_space.sample()
    action = (1.0,0.0)
    observation, reward, terminated, truncated, info = env.step(action)
    steps += 1
    if terminated or truncated:
        print(f'reset at {steps}')
        steps = 0
        observation, info = env.reset()
env.close()
