import gym
import logging
import numpy as np
import time

from sweet.agents.agent_runner import Runner
from sweet.agents.runner.stop_condition import EpisodeDoneStopCond, NstepsStopCond
from sweet.agents.dqn.dqn_agent import DqnAgent
from sweet.common.math import explained_variance


def learn(env_name='CartPole-v0'):
    # Load OpenAI Gym env
    env = gym.make(env_name)

    # Load DQN agent
    agent = DqnAgent(
        state_shape=env.observation_space.shape, 
        action_size=env.action_space.n)

    total_timesteps = 1e5
    timesteps = 0

    runner = Runner(env, agent, stop_cond=EpisodeDoneStopCond())
    tstart = time.time()     

    while timesteps < total_timesteps:
        # Collect batch of experience
        t0 = time.time()
        obs, next_obs, rewards, actions, dones, values, infos = runner.run()
        dt_xp = time.time()-t0

        # Optimize both actor and critic with gradient descent
        t0 = time.time()
        agent.memorize(zip(obs, next_obs, rewards, actions, dones, values))
        agent.decay_exploration(len(rewards))
        dt_mem = time.time()-t0

        # Update network
        t0 = time.time()
        agent.update()
        dt_update = time.time()-t0
        
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
        logging.info(f"perfo dt_xp={dt_xp} / dt_mem={dt_mem} / dt_update={dt_update}")



def learn2(
    env,
    agent,
    timesteps=1e5):
    """

    OBSOLETE


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
            total_timesteps  += 1
            rewards.append(rew)

            if done:
                sum_rewards.append(np.sum(rewards))
                logging.info("Episode {} done in {} steps / eps={}".format(nepisode, steps, agent.eps))
                nepisode += 1

        #env.render()
    
    import matplotlib.pyplot as plt
    plt.plot(sum_rewards)
    plt.show()


if __name__ == "__main__":
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
    
    env = gym.make('CartPole-v0')
    agent = DqnAgent(
        state_shape=env.observation_space.shape, 
        action_size=env.action_space.n)
    learn2(env, agent)

    #learn()
