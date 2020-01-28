import gym
import logging
import numpy as np

from sweet.agents.agent_runner import Runner
from sweet.agents.runner.stop_condition import EpisodeDoneStopCond
from sweet.agents.dqn.dqn_agent import DqnAgent


def learn(env_name='CartPole-v0'):
    # Load OpenAI Gym env
    env = gym.make(env_name)

    # Load DQN agent
    agent = DqnAgent(
        state_shape=env.observation_space.shape, 
        action_size=env.action_space.n)

    total_timesteps = 1e3
    timesteps = 0

    runner = Runner(env, agent, stop_cond=EpisodeDoneStopCond())        

    while timesteps < total_timesteps:
        # Collect batch of experience
        obs, next_obs, rewards, actions, dones, values = runner.run()

        # Optimize both actor and critic with gradient descent
        agent.memorize(zip(obs, next_obs, rewards, actions, dones, values))

        # Update network
        agent.update()

        timesteps += len(rewards)
        logging.info("timesteps = {}".format(timesteps))


if __name__ == "__main__":
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
    learn()


# def learn(
#     env,
#     agent,
#     timesteps=1e5):
#     """

#     OBSOLETE


#     Runner for RL agent: Expriment environment, memorize experiences and execute RL updates.

#     Parameters
#     ----------
#         env: gym.Env
#             OpenAI Gym environment
#         agent: sweet.agents.agent.Agent
#             RL algorithm agent
#         timesteps: int
#             Number of timesteps executed during learning
#     Returns
#     -------
#     """
#     total_timesteps = 0
#     sum_rewards = []

#     while total_timesteps < timesteps:
#         obs = env.reset()
#         done = False
#         rewards = []
#         steps = 0

#         while not done:
#             action, q_prediction = agent.act(obs)
#             if not env.action_space.contains(action):
#                 action = env.action_space.sample()

#             next_obs, rew, done, info = env.step(action)

#             # Memorize s_t, a_t, r_t, s_t+1 with a capacity N
#             agent.memorize(obs, action, rew, next_obs, done, q_prediction)

#             obs = next_obs
#             steps += 1
#             total_timesteps  += 1
#             rewards.append(rew)

#             if done:
#                 sum_rewards.append(np.sum(rewards))
#                 logging.info("Episode done in {} steps with sum rewards {}".format(steps, np.sum(rewards)))

#         agent.update()
#         #env.render()
    
#     plt.plot(sum_rewards)
#     plt.show()