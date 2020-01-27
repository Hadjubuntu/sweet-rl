
from sweet.agents.agent import Agent
from sweet.common.schedule import ConstantSchedule, LinearSchedule
from sweet.agents.a2c.a2c_actor import A2CActor
from sweet.agents.a2c.a2c_critic import A2CCritic

import numpy as np

class A2CAgent(Agent):
    """
    Simple A2C (Asynchronous Actor-Critic) implementation

    paper: https://arxiv.org/pdf/1602.01783.pdf    
    """
    def __init__(self, 
                state_shape,
                action_size,
                model='dense',
                lr=0.01,
                gamma: float=0.99):
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
        # Reshape obs expecting (nb_batch, obs_shape..) and got (obs_shape)
        obs = np.expand_dims(obs, axis=0)

        action = self.actor.act(obs)
        value = self.critic.predict(obs)

        # Proability distribution to action identifier
        action = np.argmax(action)

        return action, value

    def update(self, obs, rewards, actions, dones, values):        
        discounted_reward = self.discount_with_dones(rewards, dones, self.gamma)
        print("Shape = {}".format(obs.shape))
        print(discounted_reward)

        V = self.critic.predict(obs)
        advs = discounted_reward - V

        # Update both actor and critic
        self.critic.update(obs, values)
        self.actor.update(obs, actions, advs)