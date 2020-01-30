import tensorflow.keras.backend as K
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.losses import CategoricalCrossentropy

from sweet.models.default_models import dense
from sweet.agents.agent import Agent

# Improve TODO
import tensorflow as tf
import tensorflow.keras.layers as kl
import tensorflow.keras.losses as kls


from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Flatten


class ProbabilityDistribution(tf.keras.Model):
    def call(self, logits):
        # sample a random categorical action from given logits
        return tf.squeeze(tf.random.categorical(logits, 1), axis=-1)


def pi_model(input_shape, output_shape):
    # Create inputs
    inputs = Input(shape=input_shape)
    advs = Input(shape=1)
    x = inputs

    # Create one dense layer and one layer for output
    x = Dense(32, activation='tanh')(x)
    x = Dense(32, activation='tanh')(x)
    logits = Dense(output_shape, activation='softmax')(x)

    # Finally build model
    model = Model(inputs=[inputs, advs], outputs=logits, name="pi")
    model.summary()

    return model, advs


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
        self.model, advs = pi_model(input_shape, output_shape)
        self.loss = self.loss_func(advs)

        # Initialize model
        self.model.compile(
            loss=self.loss,
            optimizer=self.optimizer
        )
    
    def loss_func(self, _advs):
        """
        Custom loss func to add entropy


        TODOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO
        only thing missing, we didn't succeed to pass adv through model input
        see why
        """
        advs = _advs
        weighted_sparse_ce = kls.CategoricalCrossentropy(from_logits=True)

        def pi_loss(y_true, y_pred):
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

            return policy_loss

        # Return a function
        return pi_loss

    # def _logits_loss(self, acts_and_advs, logits):
    #     # a trick to input actions and advantages through same API
    #     actions, advantages = tf.split(acts_and_advs, 2, axis=-1)

    #     # sparse categorical CE loss obj that supports sample_weight arg on
    #     # call() from_logits argument ensures transformation into normalized
    #     # probabilities
    #     weighted_sparse_ce = kls.SparseCategoricalCrossentropy(
    #         from_logits=True)

    #     # policy loss is defined by policy gradients, weighted by advantages
    #     # note: we only calculate the loss on the actions we've actually taken
    #     actions = tf.cast(actions, tf.int32)
    #     policy_loss = weighted_sparse_ce(
    #         actions,
    #         logits,
    #         sample_weight=advantages
    #     )

    #     # entropy loss can be calculated via CE over itself
    #     entropy_loss = kls.categorical_crossentropy(
    #         logits, logits,
    #         from_logits=True
    #     )

    #     # here signs are flipped because optimizer minimizes
    #     return policy_loss - self.params['entropy']*entropy_loss

    def act(self, obs):
        """
        Compute current policy action
        """
        return self.tf2_fast_predict(obs)

    def update(self, obs, actions, advs):
        """
        Update actor network
        """
        loss_pi = self.tf2_fast_apply_gradients(obs, actions, advs)
        print(f"loss_pi={loss_pi}")
        return loss_pi

    # Can't eager to ops: @tf.function
    def tf2_fast_apply_gradients(self, x, y, advs):
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
