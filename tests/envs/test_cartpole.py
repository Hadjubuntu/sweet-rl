import gym
from sweet.envs.basic_env import BasicEnv


def test_cartpole_env():
    env = gym.make('CartPole-v0')
    obs = env.reset()
    print(obs)

    