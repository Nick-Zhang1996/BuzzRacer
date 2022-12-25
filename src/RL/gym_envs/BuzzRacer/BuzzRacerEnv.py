import gym
from gym import spaces
import pygame
import numpy as np
from math import radians,degrees
from copg.rcvip_simulator.VehicleModel import VehicleModel
from copg.rcvip_racing.rcvip_env_function import getRewardSingleAgent
import torch

class BuzzRacerEnv(gym.Env):
    metadata = {'render_modes':['human'],'render_fps':30}

    def __init__(self, render_mode=None):
        # TODO setup visualization

        # s(progress), d(lateral), heading, v_x, v_y, omega
        self.observation_space = spaces.Box(low=np.array([0.0,-0.3,-radians(180),-0.05,-2.0,-np.pi*2]),high=np.array([11.5,0.3,radians(180),5.0,2.0,np.pi*2]), dtype=np.float32)
        # throttle, steering
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # TODO implement visualization
        assert render_mode is None

        assert render_mode is None or render_mode in self.metadata['render_modes']
        self.render_mode = render_mode

        self.window = None
        self.clock = None

        n_batch = 1
        self.device = torch.device('cpu') # cpu is faster
        self.vehicle_model = VehicleModel(n_batch, self.device, track='rcp')


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        random = self.np_random
        s = random.uniform(0,11.4)
        d = random.uniform(-0.2,0.2)
        heading = random.uniform(-radians(20),radians(20))
        v_x = random.uniform(0,1.0)
        v_y = random.uniform(-0.5,0.5)
        omega = random.uniform(-radians(5),radians(5))

        states = (s,d,heading,v_x,v_y,omega)
        self.torch_states = torch.tensor([states],dtype=torch.float32)

        observation = states
        info = None
        if self.render_mode == 'human':
            self._render_frame()
        return observation,info

    def step(self, action):
        torch_action = torch.tensor([action],dtype=torch.float32)
        prev_state = self.torch_states
        self.torch_states = self.vehicle_model.dynModelBlendBatch(self.torch_states, torch_action)
        bounds = self.vehicle_model.getLocalBounds(self.torch_states[:,0])
        reward, done = getRewardSingleAgent(self.torch_states, bounds ,prev_state,  self.device)

        if self.render_mode == 'human':
            self._render_frame()
        observation = tuple(self.torch_states.flatten().numpy())
        reward = reward.item()
        terminated = done.item()
        truncated = False
        info = None
        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        # TODO
        raise NotImplementedError

        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # First we draw the target
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self._target_location,
                (pix_square_size, pix_square_size),
            ),
        )
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
    def close(self):
        # TODO
        return
        if self.window is not none:
            pygame.display.quit()
            pygame.quit()
