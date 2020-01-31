import gym
from gym import spaces
import numpy as np


class BasicEnv(gym.Env):
    """
    Purpose of BasicEnv is to allow unitary testing of RL algorithms
    Therefore this is dummy environment where the goal is to select action 0
    all the time :)
    """
    def __init__(self):
        super().__init__()
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(
                3,), dtype=np.float32)
        self.consecutive_rew = 0

    def reset(self):
        obs = np.random.rand(3,)
        self.consecutive_rew = 0
        return obs

    def step(self, action):
        done = False
        rew = 0

        if action == 0:
            rew = 1
            self.consecutive_rew += 1

        # Stop when agent succeed 30 times
        if self.consecutive_rew > 30:
            done = True

        # New observation
        obs = np.random.rand(3,)

        return obs, rew, done, {}
