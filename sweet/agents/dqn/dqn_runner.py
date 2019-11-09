 
import sweet.agents.dqn.dqn_agent as dqn
import gym
import numpy as np
from collections import deque
from math import log, e

def entropy(labels, base=None):
    value, counts = np.unique(labels, return_counts=True)
    norm_counts = counts / counts.sum()
    base = e if base is None else base
    
    return -(norm_counts * np.log(norm_counts)/np.log(base)).sum()


def learn(env, agent, timesteps=1e5):
    """
    Runs episode to learn
    """
    total_timesteps = 0

    while total_timesteps < timesteps:
        obs = env.reset()
        done = False
        rewards = []
        steps = 0
        act_stable = deque(maxlen=50)

        while not done:
            action, q_prediction = agent.act(obs)            
            next_obs, rew, done, info = env.step(action)

            # Tuned reward
            act_stable.append(next_obs)
            rew = np.abs(next_obs[0]) - entropy(act_stable)

            # Memorize s_t, a_t, r_t, s_t+1 with a capacity N
            agent.memorize(obs, action, rew, next_obs, done, q_prediction)

            obs = next_obs
            steps += 1
            total_timesteps  += 1
            rewards.append(rew)

            if done:
                print("Episode done in {} steps with sum rewards {}".format(steps, np.sum(rewards)))

        agent.update()
        env.render()

if __name__ == "__main__":
    env = gym.make('MountainCar-v0')

    agent = dqn.DqnAgent(state_shape=env.observation_space.shape, action_size=3)
    learn(env, agent, timesteps=1e5)
            