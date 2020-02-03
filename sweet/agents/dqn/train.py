import gym
import logging
import numpy as np
import time
from pathlib import Path

from sweet.interface.tf.tf_platform import TFPlatform
from sweet.interface.torch.torch_platform import TorchPlatform
from sweet.common.logging import init_logger
from sweet.common.time import dt_to_str
from sweet.agents.agent_runner import Runner
from sweet.agents.dqn.dqn_agent import DqnAgent
from sweet.agents.runner.stop_condition import (
    EpisodeDoneStopCond,
    NstepsStopCond
)

all_logger = init_logger()
logger = logging.getLogger("dqn-train")


def learn(
    ml_platform=TorchPlatform,
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
    log_interval: int = 1,
):
    """
    Train model with DQN agent

    Parameters
    ----------
        ml_platform: sweet.interface.MLPlatform
            Machine Learning platform (either TF2 or Torch)
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
        log_interval: int
            Timesteps frequency on which logs are printed out
            (console + tensorboard)
    """
    # Load OpenAI Gym env
    env = gym.make(env_name)

    # Load DQN agent
    agent = DqnAgent(
        ml_platform=ml_platform,
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
        expected_remaining_dt = (
            total_timesteps - timesteps
            ) / (fps + 1e-8)

        mean_episode_length = np.mean([x['steps'] for x in infos])
        mean_episode_rew = np.mean([x['rewards'] for x in infos])

        # Save model
        if (timesteps - model_checkpoint) > model_checkpoint_freq:
            agent.save_model(model_target_path)
            model_checkpoint = model_checkpoint_freq

        # Logging
        if timesteps % log_interval == 0:
            logger.info(f"Update")

            all_logger.record_tabular("total_timesteps", timesteps)
            all_logger.record_tabular("FPS", fps)
            all_logger.record_tabular("Mean rewards", mean_episode_rew)
            all_logger.record_tabular(
                "Mean episode length", mean_episode_length)
            all_logger.record_tabular("Time elapsed", dt_to_str(nseconds))
            all_logger.record_tabular("ETA", dt_to_str(expected_remaining_dt))
            all_logger.dump_tabular()


if __name__ == "__main__":
    learn()
