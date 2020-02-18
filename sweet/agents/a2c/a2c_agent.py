import numpy as np

from sweet.agents.agent import Agent
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
        lr: float or sweet.common.schedule.Schedule
            Learning rate for Actor (Policy)
        coeff_critic: float
            Ratio of critic loss compared to policy loss
        coeff_entropy: float
            Entropy factor relative to loss function
            (pg_loss + entropy_loss + value_loss)
        gamma: float
            Discount factor
    """

    def __init__(self,
                 ml_platform: MLPlatform,
                 state_shape,
                 action_size,
                 model='pi_actor_critic',
                 lr=0.003,
                 coeff_critic: float=0.5,
                 coeff_entropy: float=0.001,
                 gamma: float=0.95,
                 optimizer='adam',
                 loss='actor_categorical_crossentropy'):
        # Generic initialization
        super().__init__(
            ml_platform, lr, model, state_shape, action_size,
            optimizer=optimizer, loss=loss,
            kwargs={'coeff_vf': coeff_critic, 'coeff_entropy': coeff_entropy}
        )

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

        # As we need advs to pass in model, create a zero vector
        zero_advs = np.zeros((obs.shape[0], 1))

        # Models returns logits and given advantages
        action_logits, advs, value = self.fast_predict([obs, zero_advs])

        # So we can sample an action from that probability distribution
        action_sample_np = self.sample(action_logits).numpy()
        action = action_sample_np[0]

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
        V = self.fast_predict([obs, advs])[2]

        advs[:] = discounted_reward - V

        # Update both actor and critic

        x = [obs, advs]
        y = [actions, discounted_reward]
        loss_pi = self.fast_apply_gradients(x, y)

        # TODO retrieve loss actor and critic

        return loss_pi
