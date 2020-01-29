from abc import ABC, abstractmethod
from sweet.common.schedule import Schedule
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam

from sweet.models.default_models import dense

class Agent(ABC):
    """
    Generic class for RL algorithm
    """    
    def __init__(self, lr, model, state_shape, action_size, optimizer=Adam, loss=MeanSquaredError):
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
        self.loss = loss()
        self.model = self._build_model(model, state_shape, action_size)

        
    # [TF 2.0 error: we can't use numpy func in graph mode (eg. with tf.function)] @tf.function
    def tf2_fast_predict(self, x):
        res = self.model(x)
        return res.numpy()

    @tf.function
    def tf2_fast_apply_gradients(self, x, y):
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

        # train_loss = tf.keras.metrics.Mean(name='train_loss')
        # train_metric = tf.keras.metrics.MeanSquaredError(name='train_accuracy')

        # train_loss(loss)
        # train_metric(y, predictions)

    @abstractmethod
    def act(self, obs):
        """
        Determines action regarding current observation
        """
        pass

    def discount_with_dones(self, rewards, dones, gamma):
        """
        Compute discounted rewards
        source: OpenAI baselines (a2c.utils)
        """
        discounted = []
        r = 0
        for reward, done in zip(rewards[::-1], dones[::-1]):
            r = reward + gamma*r*(1.-done) # fixed off by one bug
            discounted.append(r)
        return np.array(discounted[::-1])

    def step_callback(self, data):
        """
        Callback function
        """
        pass

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

    def _entropy(self, labels, base=None):
        """
        Compute entry of input labels
        """
        value, counts = np.unique(labels, return_counts=True)
        norm_counts = counts / counts.sum()
        base = e if base is None else base
        
        return -(norm_counts * np.log(norm_counts)/np.log(base)).sum()

    