import gym
from gym import spaces
import numpy as np


class BasicEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(
                3,), dtype=np.float32)

    def reset(self):
        obs = np.random.rand(3,)
        return obs

    def step(self, action):
        return obs, rew, done, info
