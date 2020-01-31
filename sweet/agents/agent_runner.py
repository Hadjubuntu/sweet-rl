
import numpy as np

from sweet.agents.agent import Agent
from sweet.agents.runner.stop_condition import StopCond, EpisodeDoneStopCond


class Runner():
    def __init__(self,
                 env,
                 agent: Agent,
                 stop_cond: StopCond = EpisodeDoneStopCond(),
                 step_callback=None):
        """
        Runner aims to collect batch of experience


        Parameters
        ----------
            env: gym.Env
                Environment
            agent: sweet.agents.agent.Agent
                RL algorithm agent
            stop_cond: StopCond (default: EpisodeDoneStopCond)
                Stop condition for collection a batch of experience
            step_callaback:
                Callback function at each step function
        """
        self.env = env
        self.agent = agent
        self.stop_cond = stop_cond
        self.obs = self.env.reset()
        self.episode_steps = 0  #  TODO Better in vec_env
        self.episode_rews = 0.0
        self.step_callback = step_callback

    def run(self):
        """
        Execute the environment for nsteps to collect batch of experience
        """
        mb_obs, mb_next_obs, mb_rewards, mb_actions, mb_dones, mb_values = [], [], [], [], [], []
        mb_infos = []

        # Reset environment and stop condition
        self.stop_cond.reset()
        done = False

        # We collect until the stop condition is encountered
        while self.stop_cond.iterate(done=done):
            # Compute agent action and value (or Q-value depending on agent)
            # estimation
            action, value = self.agent.act(self.obs)

            # Take actions in env and collect experience outputs
            next_obs, rew, done, infos = self.env.step(action)
            self.episode_steps += 1
            self.episode_rews += rew

            obs_copy = np.copy(self.obs)

            # Callback (if specified)
            if self.step_callback:
                self.step_callback(
                    (obs_copy, next_obs, rew, done, action, value)
                )

            # Store all needed data
            mb_obs.append(obs_copy)
            mb_next_obs.append(next_obs)
            mb_rewards.append(rew)
            mb_actions.append(action)
            mb_dones.append(done)
            mb_values.append(value)

            self.obs = next_obs

            if done:
                infos['steps'] = self.episode_steps
                infos['rewards'] = self.episode_rews
                mb_infos.append(infos)

                self.obs = self.env.reset()
                self.episode_steps = 0
                self.episode_rews = 0.0

        # Transform list into numpy array
        mb_obs = np.asarray(mb_obs, dtype=np.float32)
        mb_next_obs = np.asarray(mb_next_obs, dtype=np.float32)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        # FIXME: int32 for action => Discrete action so far
        mb_actions = np.asarray(mb_actions, dtype=np.int32)
        mb_dones = np.asarray(mb_dones, dtype=np.int32)
        mb_values = np.asarray(mb_values, dtype=np.float32)

        return mb_obs, mb_next_obs, mb_rewards, mb_actions, mb_dones, mb_values, mb_infos
