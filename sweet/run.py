__author__ = "Adrien HADJ-SALAH"
__email__ = "adrien.hadj.salah@gmail.com"

"""
Entry-point of sweet RL.

Usage:
    python -m sweet.run -e CartPole-v0
"""

import sys
import re
import multiprocessing
import os.path as osp
import gym
import tensorflow as tf
import numpy as np

from sweet.agents.dqn.dqn_agent import DqnAgent
from sweet.agents.agent_runner import learn
import logging
import argparse

def main(args):
    """
    Create RL agent and launch training phase
    """    
    # Initialize logger
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", type=str, help="Environment to play with", default="CartPole-v0")
    args = parser.parse_args()

    env = args.e

    # Load OpenAI Gym env
    env = gym.make(env)

    # Load DQN agent
    agent = DqnAgent(state_shape=env.observation_space.shape, action_size=env.action_space.n)

    # Learn few steps
    learn(env, agent, timesteps=1e2)

if __name__ == '__main__':
    main(sys.argv)