import numpy as np

from sweet.agents.agent import Agent
from sweet.agents.a2c.a2c_actor import A2CActor
from sweet.agents.a2c.a2c_critic import A2CCritic
from sweet.interface.ml_platform import MLPlatform


class A2CAgent(Agent):
    """
    Simple A2C (Asynchronous Actor-Critic) implementation

    paper: https://arxiv.org/pdf/1602.01783.pdf

    Parameters
    ----------
        ml_platform: MLPlatform
            Instance of Machine Learning platform to use.
            TF2 = sweet.interface.tf.tf_platform.TFPlaform
            Torch = sweet.interface.tf.tf_platform.TorchPlatform
        state_shape: shape
            Observation state shape
        action_size: int
            Number of actions (Discrete only so far)
        model: Model or str
            Neural network model or string representing NN (dense, cnn)
        lr_actor: float or sweet.common.schedule.Schedule
            Learning rate for Actor (Policy)
        lr_critic: float or sweet.common.schedule.Schedule
            Learning rate for Critic (Value estimator)
        gamma: float
            Discount factor
    """

    def __init__(self,
                 ml_platform: MLPlatform,
                 state_shape,
                 action_size,
                 model='dense',
                 lr_actor=0.004,
                 lr_critic=0.002,
                 gamma: float = 0.95):
        # Generic initialization
        super().__init__(
            ml_platform, lr_actor, model, state_shape, action_size
        )

        # TODO pass model actor/critic
        self.actor = A2CActor(lr_actor, state_shape, action_size)
        self.critic = A2CCritic(lr_critic, state_shape)

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
        obs = self.encode(obs)

        action = self.actor.act(obs)
        value = self.critic.predict(obs)

        return action, value

    def update(self, obs, rewards, actions, dones, values):
        """
        Update actor and critic network with batch of datas.
        For critic, data are the discounted rewards.
        From those discounted rewards, we can compute the advantages as
        Adv(s,a) = Q(s,a) - V(s)
        For actor, data are actions taken and computed advantages.
        """
        discounted_reward = self.discount_with_dones(
            rewards, dones, self.gamma)

        # Reshape discounted_reward to have (nbatch, 1) dimension instead of
        # (nbatch,)
        discounted_reward = np.expand_dims(discounted_reward, axis=1)

        # Compute advantage / TD-error A(s,a)=Q(s,a)-V(s) (Note advantages is
        # an array of dim (nbatch, nactions))
        advs = np.zeros((len(obs), 1))
        V = self.critic.predict(obs)

        advs[:] = discounted_reward - V

        # Update both actor and critic
        loss_critic = self.critic.update(obs, discounted_reward)
        loss_actor = self.actor.update(obs, actions, advs)

        return loss_actor, loss_critic
