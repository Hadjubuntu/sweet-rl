import gym
import logging
import numpy as np
import time

from sweet.agents.agent_runner import Runner
from sweet.agents.runner.stop_condition import NstepsStopCond
from sweet.common.math import explained_variance

from sweet.agents.a2c.a2c_agent import A2CAgent


def learn(
    env_name='CartPole-v0',
    total_timesteps=1e5
):
    # Load OpenAI Gym env
    env = gym.make(env_name)

    # Load DQN agent
    agent = A2CAgent(
        state_shape=env.observation_space.shape,
        action_size=env.action_space.n)

    nenvs = 1
    nsteps = 128
    nbatch = nenvs * nsteps
    nudpates = int(total_timesteps // nbatch + 1)

    runner = Runner(env, agent, stop_cond=NstepsStopCond(nsteps))
    tstart = time.time()

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
        mean_episode_length = np.mean([x['steps'] for x in infos])
        mean_episode_rew = np.mean([x['rewards'] for x in infos])

        # Logging
        logging.info(f"Update #{nupdate}")
        logging.info(f"total_timesteps={nbatch*nupdate}")
        logging.info(f"FPS={fps}")
        logging.info(f"explained_varaince={expl_variance}")
        logging.info(f"Loss_actor={loss_actor}")
        logging.info(f"Loss_critic={loss_critic}")
        logging.info(f"Mean rewards={mean_episode_rew}")
        logging.info(f"Mean episode length={mean_episode_length}")


if __name__ == "__main__":
    logging.basicConfig(
        format='%(levelname)s:%(message)s',
        level=logging.DEBUG)

    learn()
