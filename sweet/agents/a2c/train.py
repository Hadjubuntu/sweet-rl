import gym
import numpy as np
import time
from pathlib import Path
from collections import deque

from sweet.interface.tf.tf_platform import TFPlatform
from sweet.interface.torch.torch_platform import TorchPlatform
from sweet.common.logging import Logger
from sweet.agents.agent_runner import Runner
from sweet.agents.runner.stop_condition import NstepsStopCond
from sweet.common.math import explained_variance
from sweet.agents.a2c.a2c_agent import A2CAgent
from sweet.common.time import dt_to_str
from sweet.common.utils import now_str


def learn(
    ml_platform=TorchPlatform,
    env_name='CartPole-v1',
    model='pi_actor_critic',
    total_timesteps=1e7,
    nenvs=1,
    nsteps=32,
    lr=0.001,
    coeff_critic=0.5,
    gamma: float = 0.95,
    model_checkpoint_freq: int = 1e5,
    log_interval: int = 10,
    targets: dict = {
        'output_dir': Path('./target/'),
        'models_dir': 'models_checkpoints',
        'logs_dir': 'logs',
        'tb_dir': 'tb_events'
    }
):
    """
    Train agent with A2C algorithm

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
        nenvs: int
            Number of parallel environment to collect experience
        nsteps: int
            Number of steps executed by each environment to collect experience.
            So batch size is nenvs * nsteps for each update
        lr_actor: float or sweet.common.schedule.Schedule
            Learning rate for actor
        lr_actor: float or sweet.common.schedule.Schedule
            Learning rate for critic
        gamma: float
            Discount factor
        model_checkpoint_freq: int
            Save model each "model_checkpoint_freq" update
            (so each nenvs*nsteps)
        log_interval: int
            Network update frequency on which logs are printed out
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
        "a2c-train",
        target_dir=run_target_dir,
        logs_dir=targets['logs_dir'],
        tb_dir=targets['tb_dir'])
    logger.save(run_target_dir / Path('configuration.json'), locals())

    # Load OpenAI Gym env
    env = gym.make(env_name)

    # Load DQN agent
    agent = A2CAgent(
        ml_platform=ml_platform,
        state_shape=env.observation_space.shape,
        action_size=env.action_space.n,
        model=model,
        lr=lr,
        coeff_critic=coeff_critic,
        gamma=gamma,
    )

    nbatch = nenvs * nsteps
    nudpates = int(total_timesteps // nbatch + 1)
    model_checkpoint = 0

    runner = Runner(env, agent, stop_cond=NstepsStopCond(nsteps))
    tstart = time.time()

    # Collect infos on last 10 batch runs
    u_steps, u_rewards = deque(maxlen=10), deque(maxlen=10)

    for nupdate in range(1, nudpates):
        # Collect mini-batch of experience
        obs, _, rewards, actions, dones, values, infos = runner.run()

        # Optimize both actor and critic with gradient descent
        loss_pi = agent.update(
            obs, rewards, actions, dones, values)

        # Post-processing (logging, ..)
        nseconds = time.time() - tstart
        fps = int((nupdate * nbatch) / nseconds)
        expl_variance = explained_variance(np.squeeze(values), rewards)
        expected_remaining_dt = (
            total_timesteps - (nupdate * nbatch)
            ) / (fps + 1e-8)

        steps = [x['steps'] for x in infos]
        rewards = [x['rewards'] for x in infos]

        if len(steps):
            u_steps.append(np.mean(steps))
            u_rewards.append(np.mean(rewards))

        mean_episode_length = np.mean(u_steps)
        mean_episode_rew = np.mean(u_rewards)

        # Save model
        if (nupdate - model_checkpoint) > model_checkpoint_freq:
            agent.save_model(models_dir / f'model-{nupdate}')
            model_checkpoint = model_checkpoint_freq

        # Logging
        if nupdate % log_interval == 0 or nupdate == 1:
            logger.info(f"Update #{nupdate}")

            logger.record_tabular("total_timesteps", nbatch*nupdate)
            logger.record_tabular("FPS", fps)
            logger.record_tabular("explained_varaince", expl_variance)
            logger.record_tabular("Loss_pi", loss_pi)
            logger.record_tabular("Mean rewards", mean_episode_rew)
            logger.record_tabular("Mean episode length", mean_episode_length)
            logger.record_tabular("Time elapsed", dt_to_str(nseconds))
            logger.record_tabular("ETA", dt_to_str(expected_remaining_dt))

            logger.dump_tabular()


if __name__ == "__main__":
    learn()
