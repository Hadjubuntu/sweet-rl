import gym
import logging
import numpy as np
import time
from pathlib import Path

from sweet.common.logging import init_logger
from sweet.agents.agent_runner import Runner
from sweet.agents.dqn.dqn_agent import DqnAgent
from sweet.agents.runner.stop_condition import (
    EpisodeDoneStopCond,
    NstepsStopCond
)

logger = logging.getLogger(__name__)


def learn(
    env_name='CartPole-v0',
    total_timesteps=1e5,
    lr=0.01,
    gamma: float = 0.95,
    epsilon: float = 1.0,
    epsilon_min: float = 0.01,
    epsilon_decay: float = 0.995,
    replay_buffer: int = 2000,
    model_target_path: Path = Path('./target/model.h5'),
    model_checkpoint_freq: int = 1e3,
):
    """
    Train model with DQN agent

    Parameters
    ----------
        env_name: str
            Name of OpenAI Gym environment
        total_timesteps: int
            Number of training steps
        lr: float or sweet.common.schedule.Schedule
            Learning rate
        gamma: float
            Discount factor
        epsilon: float
            Exploration probability
            (choose random action over max Q-value action
        epsilon_min: float
            Minimum probability of exploration
        epsilon_decay: float
            Decay of exploration at each update
        replay_buffer: int
            Size of the  replay buffer
        model_target_path: Path
            Path to the model in order to save while learning
            (.h5 extension needed)
        model_checkpoint_freq: int
            Save model each "model_checkpoint_freq" steps
    """
    # Load OpenAI Gym env
    env = gym.make(env_name)

    # Load DQN agent
    agent = DqnAgent(
        state_shape=env.observation_space.shape,
        action_size=env.action_space.n,
        model='dense',
        lr=lr,
        gamma=gamma,
        epsilon=epsilon, epsilon_min=epsilon_min, epsilon_decay=epsilon_decay,
        replay_buffer=replay_buffer)

    # Variables to monitor
    timesteps = 0
    model_checkpoint = 0

    callback = agent.step_callback

    runner = Runner(
        env,
        agent,
        stop_cond=EpisodeDoneStopCond(),
        step_callback=callback
    )

    tstart = time.time()

    # Iterate act/memorize/update until total_timesteps is reached
    while timesteps < total_timesteps:

        # Execute runner to collect batch of experience
        obs, next_obs, rewards, actions, dones, values, infos = runner.run()

        """
        Notes
        -----
            Memorize experience + network update are done within the run
            method through callback function. It has be done that way to have
            a single runner for both algorithms updating their network
            at each step, and those updating after nsteps
        """

        # Post-processing (logging, ..)
        nseconds = time.time() - tstart
        timesteps += len(rewards)
        fps = int(timesteps / nseconds)

        mean_episode_length = np.mean([x['steps'] for x in infos])
        mean_episode_rew = np.mean([x['rewards'] for x in infos])

        # Save model
        if (timesteps - model_checkpoint) > model_checkpoint_freq:
            agent.save_model(model_target_path)
            model_checkpoint = model_checkpoint_freq

        # Logging
        logger.info(f"Update")
        logger.info(f"total_timesteps={timesteps}")
        logger.info(f"FPS={fps}")
        logger.info(f"Mean rewards={mean_episode_rew}")
        logger.info(f"Mean episode length={mean_episode_length}")


if __name__ == "__main__":
    init_logger()

    learn()
