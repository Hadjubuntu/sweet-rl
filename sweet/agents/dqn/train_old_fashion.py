"""
If you find the method learn from train.py a bit harsch with its callback
and masked optimization, just see this method which is written
in old-fashion way
"""
import gym
import logging
import numpy as np

from sweet.common.logging import init_logger
from sweet.agents.dqn.dqn_agent import DqnAgent
from sweet.agents.runner.stop_condition import (
    EpisodeDoneStopCond, NstepsStopCond
)

logger = logging.getLogger(__name__)


def learn_oldfashion(
        env,
        agent,
        timesteps=1e5):
    """
    Old-fashion runner for RL agent: Expriment environment,
    memorize experiences and execute RL updates.

    Parameters
    ----------
        env: gym.Env
            OpenAI Gym environment
        agent: sweet.agents.agent.Agent
            RL algorithm agent
        timesteps: int
            Number of timesteps executed during learning
    Returns
    -------
    """
    total_timesteps = 0
    nepisode = 1
    sum_rewards = []

    while total_timesteps < timesteps:
        obs = env.reset()
        done = False
        rewards = []
        steps = 0

        while not done:
            action, q_prediction = agent.act(obs)

            next_obs, rew, done, info = env.step(action)

            # Memorize s_t, a_t, r_t, s_t+1 with a capacity N
            agent.memorize([(obs, next_obs, rew, action, done, q_prediction)])
            agent.decay_exploration(1)

            agent.update()

            obs = next_obs
            steps += 1
            total_timesteps += 1
            rewards.append(rew)

            if done:
                sum_rewards.append(np.sum(rewards))
                logger.info(
                    "Episode {} done in {} steps / eps={}".format(
                        nepisode, steps, agent.eps)
                )
                nepisode += 1

        # env.render()

    import matplotlib.pyplot as plt
    plt.plot(sum_rewards)
    plt.show()


if __name__ == "__main__":
    init_logger()

    env = gym.make('CartPole-v0')
    agent = DqnAgent(
        state_shape=env.observation_space.shape,
        action_space=env.action_space)

    learn_oldfashion(env, agent)
