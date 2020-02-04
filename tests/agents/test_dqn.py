import logging

import gym
from sweet.agents.dqn.dqn_agent import DqnAgent
from sweet.agents.dqn.train import learn
from sweet.common.logging import Logger
from sweet.envs.basic_env import BasicEnv

logger = Logger(log_in_file=False)


def test_dqn(env_name='basic-v0'):
    """
    Test DQN on a dummy env
    Note that CartPole-V0 is fast enough for this test
    """

    # Learn few steps
    learn(
        env_name=env_name,
        total_timesteps=10,
        lr=0.01,
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        replay_buffer=2000,
    )

    # TODO assert
