# TEMPORARY: shall be capitalized into sweet.agents.agent_runner
from sweet.agents.a2c.a2c_agent import A2CAgent

import gym
import logging
import numpy as np

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

class Runner():
    def __init__(self, env, nsteps, agent):
        self.env = env
        self.nsteps = nsteps
        self.agent = agent

    def run(self):
        mb_obs, mb_rewards, mb_actions, mb_dones, mb_values = [], [], [], [], []

        obs = self.env.reset()

        for _ in range(self.nsteps):
            # Compute agent action and value estimation
            action, value = self.agent.act(obs)

            # Take actions in env and collect experience outputs
            obs, rew, done, info = self.env.step(action)

            mb_obs.append(obs)
            mb_rewards.append(rew)
            mb_actions.append(action)
            mb_dones.append(done)
            mb_values.append(value)

            if done:
                obs = self.env.reset()

        mb_obs = np.asarray(mb_obs, dtype=np.float32)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.float32)
        mb_values = np.asarray(mb_values, dtype=np.float32)

        return mb_obs, mb_rewards, mb_actions, mb_dones, mb_values

def exec_runner(env_name='CartPole-v0'):
    # Load OpenAI Gym env
    env = gym.make(env_name)

    # Load DQN agent
    agent = A2CAgent(
        state_shape=env.observation_space.shape, 
        action_size=env.action_space.n)

    total_timesteps = 1e3
    nenvs = 1
    nsteps = 128
    nbatch = nenvs * nsteps
    nudpates = int(total_timesteps//nbatch+1)

    runner = Runner(env, nsteps, agent)        

    for nupdate in range(1, nudpates):
        # Collect mini-batch of experience
        obs, rewards, actions, dones, values = runner.run()

        # Optimize both actor and critic with gradient descent
        agent.update(obs, rewards, actions, dones, values)


if __name__ == "__main__":
    exec_runner()