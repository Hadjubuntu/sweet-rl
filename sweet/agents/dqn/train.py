import gym
import numpy as np
import time
from pathlib import Path

from sweet.interface.torch.torch_platform import TorchPlatform
from sweet.common.logging import Logger
from sweet.common.time import dt_to_str
from sweet.agents.agent_runner import Runner
from sweet.agents.dqn.dqn_agent import DqnAgent
from sweet.agents.runner.stop_condition import (
    EpisodeDoneStopCond,
    NstepsStopCond
)
from sweet.common.utils import now_str


def learn(
    ml_platform=TorchPlatform,
    env_name='CartPole-v0',
    model='dense',
    total_timesteps=1e5,
    lr=0.01,
    gamma: float = 0.95,
    epsilon: float = 1.0,
    epsilon_min: float = 0.01,
    epsilon_decay: float = 0.995,
    replay_buffer: int = 2000,
    model_checkpoint_freq: int = 1e5,
    log_interval: int = 1,
    targets: dict = {
        'output_dir': Path('./target/'),
        'models_dir': 'models_checkpoints',
        'logs_dir': 'logs',
        'tb_dir': 'tb_events'
    }
):
    """
    Train model with DQN agent

    Parameters
    ----------
        ml_platform: sweet.interface.MLPlatform
            Machine Learning platform (either TF2 or Torch)
        env_name: str
            Name of OpenAI Gym environment
        model: str or model from ML platform
            Neural network or string describing neural network
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
        model_checkpoint_freq: int
            Save model each "model_checkpoint_freq" steps
        log_interval: int
            Timesteps frequency on which logs are printed out
            (console + tensorboard)
        targets: dict
            Output directories:
                output_dir: Main target directory
                run_format_dir: Current run target directory
                models_dir: Models checkpoint directory
                logs_dir: Logs directory
                tb_dir: TensorBoard event files directory


    """
    # Target paths and save conf
    run_target_dir = targets['output_dir'] / f"run_{now_str()}"
    models_dir = run_target_dir / targets['models_dir']

    logger = Logger(
        "dqn-train",
        target_dir=run_target_dir,
        logs_dir=targets['logs_dir'],
        tb_dir=targets['tb_dir'])
    logger.save(run_target_dir / Path('configuration.json'), locals())

    # Load OpenAI Gym env
    env = gym.make(env_name)

    # Load DQN agent
    agent = DqnAgent(
        ml_platform=ml_platform,
        state_shape=env.observation_space.shape,
        action_size=env.action_space.n,
        model=model,
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
            agent.save_model(models_dir / f'model-{timesteps}')
            model_checkpoint = model_checkpoint_freq

        # Logging
        if timesteps % log_interval == 0:
            logger.info(f"Update")

            logger.record_tabular("total_timesteps", timesteps)
            logger.record_tabular("FPS", fps)
            logger.record_tabular("Mean rewards", mean_episode_rew)
            logger.record_tabular(
                "Mean episode length", mean_episode_length)
            logger.record_tabular("Time elapsed", dt_to_str(nseconds))
            logger.record_tabular("ETA", dt_to_str(expected_remaining_dt))
            logger.dump_tabular()


if __name__ == "__main__":
    learn()
