import gym
from sweet.envs.basic_env import BasicEnv


def test_basic_env():
    env = gym.make('basic-v0')
    obs = env.reset()
    print(obs)