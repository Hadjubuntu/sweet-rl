import gym
from sweet.agents.a2c.a2c_agent import A2CAgent
from sweet.agents.agent_runner import learn
import logging

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)


def dqn(env_name='CartPole-v0'):
    # Load OpenAI Gym env
    env = gym.make(env_name)

    # Load DQN agent
    agent = A2CAgent(
        state_shape=env.observation_space.shape, 
        action_size=env.action_space.n)

    # Learn few steps
    learn(env, agent, timesteps=1e4)
            


if __name__ == "__main__":
    dqn()