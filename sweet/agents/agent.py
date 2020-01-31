from abc import ABC, abstractmethod
from sweet.common.schedule import Schedule
import numpy as np

import tensorflow as tf
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Mean
from sweet.models.default_models import dense


class Agent(ABC):
    """
    Generic class for RL algorithm
    """

    def __init__(
            self,
            lr,
            model,
            state_shape,
            action_size,
            optimizer=Adam,
            loss=MeanSquaredError):
        """
        Generic initialization of RL algorithm
        """
        # Input/output shapes
        self.state_shape = state_shape
        self.action_size = action_size

        # Set common hyperparameters
        self.lr = lr

        # Initialize model/optimizer and loss
        self.optimizer = optimizer(lr=self._lr())
        self.eval_loss = Mean('loss')

        self.loss, self.model = None, None
        if loss:
            self.loss = loss()
        if model:
            self.model = self._build_model(model, state_shape, action_size)

    def sample(self, logits):
        """
        Sample distribution
        """
        return tf.squeeze(tf.random.categorical(logits, 1), axis=-1)

    def encode(self, var):
        """
        From raw observation to encoded obs for neural network application
        """
        var = np.expand_dims(var, axis=0)
        return var.astype(np.float32)

    def save_model(self, target_path):
        """
        Save model to HDF5 file (.h5 extension needed)
        """
        target_path = str(target_path)

        if not target_path.endswith('.h5'):
            target_path = f"{target_path}.h5"

        self.model.save(target_path)

    @tf.function
    def _graph_predict(self, x):
        return self.model(x)

    def fast_predict(self, x):
        """
        Model prediction against observation x
        """
        # [TF 2.0 error: we can't use numpy func in graph mode
        # (eg. with tf.function)] @tf.function, this is why we call a
        # sub-function
        res = self._graph_predict(x)

        # Convert from tensor to numpy array
        if isinstance(res, list):
            for idx, element in enumerate(res):
                res[idx] = element.numpy()
        else:
            res = res.numpy()

        return res

    @tf.function
    def fast_apply_gradients(self, x, y):
        """
        This is a TensorFlow function, run once for each epoch for the
        whole input. We move forward first, then calculate gradients
        with Gradient Tape to move backwards.
        """
        with tf.GradientTape() as tape:
            predictions = self.model(x)
            loss = self.loss(y, predictions)

        trainable_vars = self.model.trainable_weights

        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        return self.eval_loss(loss)

    @abstractmethod
    def act(self, obs):
        """
        Determines action regarding current observation
        """

    def discount_with_dones(self, rewards, dones, gamma):
        """
        Compute discounted rewards
        source: OpenAI baselines (a2c.utils)
        """
        discounted = []
        r = 0
        for reward, done in zip(rewards[::-1], dones[::-1]):
            r = reward + gamma * r * (1. - done)  # fixed off by one bug
            discounted.append(r)
        return np.array(discounted[::-1])

    def step_callback(self, data):
        """
        Callback function
        """

    def _lr(self, t=0):
        """
        Learning rate computation
        """
        if isinstance(self.lr, float):
            return self.lr
        elif isinstance(self.lr, Schedule):
            return self.lr.value(t)
        else:
            raise NotImplementedError()

    def _build_model(self, model='dense', state_shape=None, action_shape=None):
        """
        Build model from TF-Keras model or string
        """
        if isinstance(model, str):
            model = dense(input_shape=state_shape, output_shape=action_shape)

        model.compile(loss=self.loss, optimizer=self.optimizer)

        return model
