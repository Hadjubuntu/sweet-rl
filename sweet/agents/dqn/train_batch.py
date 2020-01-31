import gym
import logging
import numpy as np
import time
import tensorflow as tf

from sweet.common.logging import init_logger
from sweet.agents.agent_runner import Runner
from sweet.agents.runner.stop_condition import (
    EpisodeDoneStopCond, NstepsStopCond
)
from sweet.agents.dqn.dqn_agent import DqnAgent
from sweet.common.math import explained_variance

logger = logging.getLogger(__name__)


def train_dqn_batch(env_name='CartPole-v0'):
    """
    Faster version of DQN train classical method:
    Here we collect batch of data before updating neural network.
    Downside: convergence is less stable


    FIXME: doesn't work so far
    """
    # Load OpenAI Gym env
    env = gym.make(env_name)

    # Load DQN agent
    agent = DqnAgent(
        state_shape=env.observation_space.shape,
        action_size=env.action_space.n,
        replay_buffer=50000)

    total_timesteps = 1e5
    timesteps = 0

    runner = Runner(env, agent, stop_cond=NstepsStopCond(32))
    tstart = time.time()

    while timesteps < total_timesteps:
        # Collect batch of experience
        obs, next_obs, rewards, actions, dones, values, infos = runner.run()

        # Note: Memorize experience + network update are done in callback fn
        agent.memorize(zip(obs, next_obs, rewards, actions, dones, values))

        # update network
        agent.update(batch_size=32)

        # Post-processing (logging, ..)
        nseconds = time.time() - tstart
        timesteps += len(rewards)
        fps = int(timesteps / nseconds)

        mean_episode_length = np.mean([x['steps'] for x in infos])
        mean_episode_rew = np.mean([x['rewards'] for x in infos])

        # Logging
        logger.info(f"Update")
        logger.info(f"total_timesteps={timesteps}")
        logger.info(f"FPS={fps}")
        logger.info(f"Mean rewards={mean_episode_rew}")
        logger.info(f"Mean episode length={mean_episode_length}")


if __name__ == "__main__":
    init_logger()

    train_dqn_batch()
