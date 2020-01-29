
from sweet.agents.agent import Agent
from sweet.common.schedule import ConstantSchedule, LinearSchedule
from sweet.agents.a2c.a2c_actor import A2CActor
from sweet.agents.a2c.a2c_critic import A2CCritic

import numpy as np
import logging


class A2CAgent(Agent):
    """
    Simple A2C (Asynchronous Actor-Critic) implementation

    paper: https://arxiv.org/pdf/1602.01783.pdf

    Parameters
    ----------
        state_shape: shape
            Observation state shape
        action_size: int
            Number of actions (Discrete only so far)
        model: Model or str
            Neural network model or string representing NN (dense, cnn)
        lr: float or sweet.common.schedule.Schedule
            Learning rate
        gamma: float
            Discount factor
    """

    def __init__(self,
                 state_shape,
                 action_size,
                 model='dense',
                 lr=0.001,
                 gamma: float = 0.95):
        # Generic initialization
        super().__init__(lr, model, state_shape, action_size)

        # TODO pass model actor/critic
        self.actor = A2CActor(lr, state_shape, action_size)
        self.critic = A2CCritic(lr, state_shape)

        # Input/output shapes
        self.state_shape = state_shape
        self.action_size = action_size

        # Hyperparameters
        self.gamma = gamma

    def act(self, obs):
        """
        Execute actor to get action and critic to estimate current state value

        Parameters
        ----------
            obs: spaces
                Observation space from environment

        Returns
        ----------
            action:
                Current action chosen
            value: float
                Estimated value from critic
        """
        # Reshape obs expecting (nb_batch, obs_shape..) and got (obs_shape)
        obs = np.expand_dims(obs, axis=0)

        action = self.actor.act(obs)
        value = self.critic.predict(obs)

        # Proability distribution to action identifier
        action = np.argmax(action)

        return action, value

    def update(self, obs, rewards, actions, dones, values):
        """
        Update actor and critic network with batch of datas
        """
        discounted_reward = self.discount_with_dones(
            rewards, dones, self.gamma)

        # Reshape discounted_reward to have (nbatch, 1) dimension instead of
        # (nbatch,)
        discounted_reward = np.expand_dims(discounted_reward, axis=1)

        # Compute advantage / TD-error A(s,a)=Q(s,a)-V(s) (Note advantages is
        # an array of dim (nbatch, nactions))
        advs = np.zeros((len(obs), self.action_size))
        V = self.critic.predict(obs)
        actions_indexes = actions.astype(np.int32)
        advs[:, actions_indexes] = discounted_reward - V

        # Update both actor and critic
        loss_critic = self.critic.update(obs, discounted_reward)
        loss_actor = self.actor.update(obs, actions, advs)

        return loss_actor, loss_critic
