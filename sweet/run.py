import sys
import re
import multiprocessing
import os.path as osp
import gym
import numpy as np
from pathlib import Path

from sweet.agents.dqn.train import learn as dqn_train
from sweet.agents.a2c.train import learn as a2c_train

from sweet.interface.tf.tf_platform import TFPlatform
from sweet.interface.torch.torch_platform import TorchPlatform
from sweet.common.logging import Logger

import argparse

__author__ = "Adrien HADJ-SALAH"
__email__ = "adrien.hadj.salah@gmail.com"

"""
Entry-point of sweet RL.

Example:
    python -m sweet.run --env CartPole-v0 --algo=dqn
"""


def make_agent_train_func(agent_str):
    """
    Build RL agent from string
    """
    if agent_str == 'dqn':
        return dqn_train
    elif agent_str == 'a2c':
        return a2c_train
    else:
        raise NotImplementedError(f'Unknow agent: {agent_str}')


def make_ml_platform(ml_platform_str):
    """
    Build ML platfrom from string
    """
    if ml_platform_str == 'tf':
        return TFPlatform
    elif ml_platform_str == 'torch':
        return TorchPlatform
    else:
        raise NotImplementedError(f'Unknow ML platform: {ml_platform_str}')


def run(
    agent_str, ml_platform_str, env_str, model_str, timesteps, output_dir_str
):
    """
    Create RL agent and launch training phase
    """
    # Quick-fix for A2C, force model to be compatible with I/O:
    if agent_str == 'a2c':
        model_str = 'pi_actor_critic'

    # Create RL agent
    agent_train_func = make_agent_train_func(agent_str)

    # Execute agent training
    agent_train_func(
        ml_platform=make_ml_platform(ml_platform_str),
        env_name=env_str,
        model=model_str,
        total_timesteps=timesteps,
        lr=0.001,
        targets={
            'output_dir': Path(output_dir_str),
            'models_dir': 'models_checkpoints',
            'logs_dir': 'logs',
            'tb_dir': 'tb_events'
        }
    )


def parse_arguments():
    # Initialize logger
    logger = Logger("sweet-run", log_in_file=False)

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env",
        type=str,
        help="Environment to play with",
        default="CartPole-v0")
    parser.add_argument(
        "--algo",
        type=str,
        help="RL agent",
        default="dqn")
    parser.add_argument(
        "--ml",
        type=str,
        help="ML platform (tf or torch)",
        default="tf")
    parser.add_argument(
        "--model",
        type=str,
        help="Model (dense, pi_actor_critic)",
        default="dense")
    parser.add_argument(
        "--timesteps",
        type=int,
        help="Number of training steps",
        default=int(1e2))
    parser.add_argument(
        "--output",
        type=str,
        help="Output directory (eg. './target/')",
        default="./target/")
    args = parser.parse_args()

    # Build variables from arguments
    env_str = args.env
    agent_str = args.algo
    ml_platform_str = args.ml
    model_str = args.model
    timesteps = int(args.timesteps)
    output_dir_str = args.output

    return (
        agent_str,
        ml_platform_str,
        env_str,
        model_str,
        timesteps,
        output_dir_str
    )


if __name__ == '__main__':
    arguments = parse_arguments()
    run(*arguments)
