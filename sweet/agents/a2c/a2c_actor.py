import tensorflow.keras.backend as K
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.losses import CategoricalCrossentropy

from sweet.models.default_models import dense
from sweet.agents.agent import Agent

# Improve TODO
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as kl
import tensorflow.keras.losses as kls


from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Flatten, Conv2D


def pi_model(input_shape, output_shape):
    # Create inputs
    inputs = Input(shape=input_shape)
    advs = Input(shape=1)
    x = inputs

    # Create one dense layer and one layer for output
    x = Dense(8, activation='relu')(x)
    x = Flatten()(x)
    logits = Dense(output_shape)(x)

    # Finally build model
    model = Model(inputs=[inputs, advs], outputs=[logits, advs], name="pi")
    model.summary()

    return model


class A2CActor(Agent):
    def __init__(self, lr, input_shape, output_shape):
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
            lr,
            None,
            input_shape,
            output_shape,
            optimizer=Adam,
            loss=None
        )

        # Output of this model are logits (log probability)
        self.model = pi_model(input_shape, output_shape)
        self.loss = self.loss_func(_entropy_coeff=0.0001)

        # Initialize model
        self.model.compile(
            loss=self.loss,
            optimizer=self.optimizer
        )

    def loss_func(self, _entropy_coeff=0.0001):
        """
        Loss for actor: policy loss + entropy
        """
        #  FIXME forced to use sparse categorical crossentropy due to y_true
        # coming from actions without one-hot encoding
        weighted_sparse_ce = kls.SparseCategoricalCrossentropy(
            from_logits=True)
        entropy_coeff = _entropy_coeff

        def pi_loss(y_true, y_pred_and_advs):
            y_pred, advs = y_pred_and_advs[0], y_pred_and_advs[1]
            # First, one-hot encoding of true value y_true
            y_true = tf.expand_dims(tf.cast(y_true, tf.int32), axis=1)
            # No-need ? y_true = tf.one_hot(, self.action_size)

            # Execute categorical crossentropy
            neglogp = weighted_sparse_ce(
                y_true,  # True actions chosen
                y_pred,  # Logits from model
                # sample_weight=advs
            )
            policy_loss = tf.reduce_mean(advs * neglogp)

            entropy_loss = kls.categorical_crossentropy(
                y_pred, y_pred,
                from_logits=True
            )

            return policy_loss - entropy_coeff * entropy_loss

        # Return a function
        return pi_loss

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
        loss_pi = self._apply_gradients(obs, actions, advs)
        return loss_pi

    @tf.function
    def _apply_gradients(self, x, y, advs):
        """
        CUSTOM for actor
        """
        with tf.GradientTape() as tape:
            predictions = self.model([x, advs])
            loss = self.loss(y, predictions)

        trainable_vars = self.model.trainable_weights

        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        return self.eval_loss(loss)
