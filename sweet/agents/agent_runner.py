 
import gym
import numpy as np
from collections import deque
from math import log, e
import logging

import matplotlib.pyplot as plt


def learn(
    env,
    agent,
    timesteps=1e5):
    """
    Runner for RL agent: Expriment environment, memorize experiences and execute RL updates.

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
    sum_rewards = []

    while total_timesteps < timesteps:
        obs = env.reset()
        done = False
        rewards = []
        steps = 0

        while not done:
            action, q_prediction = agent.act(obs)
            if not env.action_space.contains(action):
                action = env.action_space.sample()

            next_obs, rew, done, info = env.step(action)

            # Memorize s_t, a_t, r_t, s_t+1 with a capacity N
            agent.memorize(obs, action, rew, next_obs, done, q_prediction)

            obs = next_obs
            steps += 1
            total_timesteps  += 1
            rewards.append(rew)

            if done:
                sum_rewards.append(np.sum(rewards))
                logging.info("Episode done in {} steps with sum rewards {}".format(steps, np.sum(rewards)))

        agent.update()
        #env.render()
    
    plt.plot(sum_rewards)
    plt.show()