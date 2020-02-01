

from sweet.agents.agent import Agent
from sweet.interface.ml_platform import MLPlatform

import numpy as np


class A2CActor(Agent):
    def __init__(
        self,
        ml_platform: MLPlatform,
        lr,
        input_shape,
        output_shape
    ):
        """
        Initialize critic network of A2C

        Parameters
        ----------
            lr: float or sweet.common.schedule.Schedule
                Learning rate
            input_shape: shape
                Observation state shape
            output_shape: int
                Number of actions (Discrete only so far)
        """

        super().__init__(
            ml_platform=ml_platform,
            lr=lr,
            model='pi_actor',
            state_shape=input_shape,
            action_size=output_shape,
            optimizer='adam',
            loss='actor_categorical_crossentropy'
        ) 

    def act(self, obs):
        """
        Compute current policy action
        """
        # As we need advs to pass in model, create a zero vector
        zero_advs = np.zeros((obs.shape[0], 1))

        # Models returns logits and given advantages
        action_logits, advs = self.fast_predict([obs, zero_advs])

        # So we can sample an action from that probability distribution
        action_sample_np = self.sample(action_logits).numpy()
        action = action_sample_np[0]

        return action

    def update(self, obs, actions, advs):
        """
        Update actor network
        """
        # Input of NN are observation and advantages
        x = [obs, advs]
        loss_pi = self.fast_apply_gradients(x, actions)
        return loss_pi

    # @tf.function
    # def _apply_gradients(self, x, y, advs):
    #     """
    #     CUSTOM for actor
    #     """
    #     with tf.GradientTape() as tape:
    #         predictions = self.model([x, advs])
    #         loss = self.loss(y, predictions)

    #     trainable_vars = self.model.trainable_weights

    #     gradients = tape.gradient(loss, trainable_vars)
    #     self.optimizer.apply_gradients(zip(gradients, trainable_vars))

    #     return self.eval_loss(loss)
