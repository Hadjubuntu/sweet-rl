import gym
import logging
import numpy as np

from sweet.agents.agent_runner import Runner
from sweet.agents.runner.stop_condition import NstepsStopCond

# TEMPORARY: shall be capitalized into sweet.agents.agent_runner
from sweet.agents.a2c.a2c_agent import A2CAgent


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

    runner = Runner(env, agent, stop_cond=NstepsStopCond(nsteps))        

    for _ in range(1, nudpates):
        # Collect mini-batch of experience
        obs, _, rewards, actions, dones, values = runner.run()

        # Optimize both actor and critic with gradient descent
        agent.update(obs, rewards, actions, dones, values)


if __name__ == "__main__":
    exec_runner()