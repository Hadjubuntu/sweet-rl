import gym
import logging
import numpy as np
import time
from pathlib import Path
from collections import deque

from sweet.interface.tf_platform import TFPlatform
from sweet.common.logging import init_logger
from sweet.agents.agent_runner import Runner
from sweet.agents.runner.stop_condition import NstepsStopCond
from sweet.common.math import explained_variance
from sweet.agents.a2c.a2c_agent import A2CAgent


logger = logging.getLogger("a2c-train")


def learn(
    ml_platform=TFPlatform,
    env_name='CartPole-v0',
    total_timesteps=1e5,
    nenvs=1,
    nsteps=32,
    lr_actor=0.004,
    lr_critic=0.002,
    gamma: float = 0.95,
    model_target_path: Path = Path('./target/model.h5'),
    model_checkpoint_freq: int = 50,
):
    """
    Train agent with A2C algorithm

    Parameters
    ----------
        ml_platform: sweet.interface.MLPlatform
            Machine Learning platform (either TF2 or Torch)
        env_name: str
            Name of OpenAI Gym environment
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
        model_target_path: Path
            Path to the model in order to save while learning
            (.h5 extension needed)
        model_checkpoint_freq: int
            Save model each "model_checkpoint_freq" update
            (so each nenvs*nsteps)
    """
    # Load OpenAI Gym env
    env = gym.make(env_name)

    # Load DQN agent
    agent = A2CAgent(
        ml_platform=ml_platform,
        state_shape=env.observation_space.shape,
        action_size=env.action_space.n,
        model='dense',
        lr_actor=0.004,
        lr_critic=0.002,
        gamma=0.95
    )

    nenvs = 1
    nsteps = 32
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
        loss_actor, loss_critic = agent.update(
            obs, rewards, actions, dones, values)

        # Post-processing (logging, ..)
        nseconds = time.time() - tstart
        fps = int((nupdate * nbatch) / nseconds)
        expl_variance = explained_variance(np.squeeze(values), rewards)

        steps = [x['steps'] for x in infos]
        rewards = [x['rewards'] for x in infos]

        if len(steps):
            u_steps.append(np.mean(steps))
            u_rewards.append(np.mean(rewards))

        mean_episode_length = np.mean(u_steps)
        mean_episode_rew = np.mean(u_rewards)

        # Save model
        if (nupdate - model_checkpoint) > model_checkpoint_freq:
            agent.save_model(model_target_path)
            model_checkpoint = model_checkpoint_freq

        # Logging
        logger.info(f"Update #{nupdate}")
        logger.info(f"total_timesteps={nbatch*nupdate}")
        logger.info(f"FPS={fps}")
        logger.info(f"explained_varaince={expl_variance}")
        logger.info(f"Loss_actor={loss_actor}")
        logger.info(f"Loss_critic={loss_critic}")
        logger.info(f"Mean rewards={mean_episode_rew}")
        logger.info(f"Mean episode length={mean_episode_length}")


if __name__ == "__main__":
    init_logger()

    learn()
