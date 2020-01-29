import gym
import logging
import numpy as np
import time
import tensorflow as tf

from sweet.agents.agent_runner import Runner
from sweet.agents.runner.stop_condition import (
    EpisodeDoneStopCond, 
    NstepsStopCond
)
from sweet.agents.dqn.dqn_agent import DqnAgent


# TODO: add parameters to learning method with default values
def learn(env_name='CartPole-v0'):
    # Load OpenAI Gym env
    env = gym.make(env_name)

    # Load DQN agent
    agent = DqnAgent(
        state_shape=env.observation_space.shape,
        action_size=env.action_space.n)

    total_timesteps = 1e5
    timesteps = 0

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
        nseconds = time.time()-tstart
        timesteps += len(rewards)
        fps = int(timesteps/nseconds)
        
        mean_episode_length = np.mean([x['steps'] for x in infos])
        mean_episode_rew = np.mean([x['rewards'] for x in infos])

        # Logging
        logging.info(f"Update")
        logging.info(f"total_timesteps={timesteps}")
        logging.info(f"FPS={fps}")
        logging.info(f"Mean rewards={mean_episode_rew}")
        logging.info(f"Mean episode length={mean_episode_length}")

if __name__ == "__main__":
    logging.basicConfig(
        format='%(levelname)s:%(message)s',
        level=logging.DEBUG)

    learn()
