from abc import ABC, abstractmethod

import tensorflow as tf
from tensorflow.keras.layers import (
    Dense, Conv2D, Input, LSTM, Embedding, Dropout, Activation, Flatten
)
import tensorflow.keras.losses as kls

import numpy as np


class TFDistribution:
    """
    Generic class for distribution
    """
    def __init__(self):
        pass

    @abstractmethod
    def pd_from_latent(self):
        pass

    @abstractmethod
    def entropy(self, x):
        pass

    @abstractmethod
    def sample(self):
        pass

    @abstractmethod
    def neglogp(self, x_true, x):
        pass

    @abstractmethod
    def n(self):
        pass


class TFCategoricalDist(TFDistribution):
    def __init__(self, n_cat):
        super().__init__()
        self.n_cat = n_cat
        self.cross_entrop = kls.CategoricalCrossentropy(
            from_logits=True)

    def pd_from_latent(self, x_latent):
        logits = Dense(self.n_cat)(x_latent)
        return logits

    def neglogp(self, x_true, x):
        # First, one-hot encoding of true value y_true
        x_true = tf.expand_dims(
            tf.cast(x_true, tf.int32),
            axis=1
        )
        x_true = tf.one_hot(x_true, depth=self.n_cat)

        return self.cross_entrop(x_true, x)

    def n(self):
        return self.n_cat

    def entropy(self, x):
        return self.cross_entrop(x, x)

    def sample(self, logits):
        # Sample distribution from logits
        return tf.squeeze(tf.random.categorical(logits, 1), axis=-1)


class TFDiagGaussianDist(TFDistribution):
    def __init__(self, size):
        super().__init__()
        self.size = size
        self.mean = None
        self.logstd = None

    def pd_from_latent(self, x_latent):
        self.mean = Dense(self.size)(x_latent)
        self.logstd = Dense(self.size, kernel_initializer='zeros')(x_latent)

        return tf.concat([self.mean, self.logstd], axis=1)

    def neglogp(self, x_true, x):
        if len(x.shape) > 1:
            mean, logstd = tf.split(x, 2, axis=1)

            # Trial direct
            # nlaw = tf.compat.v1.distributions.Normal(mean, logstd)
            # return -tf.math.log(nlaw.prob(x_true) + 1e-5)

            # FIXME: neglogp lead to divergence of mean, logstd layer
            std = tf.exp(logstd)

            return 0.5 * tf.reduce_sum(
                tf.square((x_true - mean) / std),
                axis=-1) \
                + 0.5 * np.log(2.0 * np.pi)  \
                + tf.reduce_sum(logstd, axis=-1)            
        else:
            return 0.0

    def n(self):
        return 2 * self.size

    def entropy(self, x):
        if len(x.shape) > 1:
            _, std = tf.split(x, 2, axis=1)

            return tf.reduce_sum(
                std + .5 * np.log(2.0 * np.pi * np.e),
                axis=-1
            )
        else:
            return 0.0

    def sample(self, logits):
        if len(logits.shape) > 1:
            mean, logstd = tf.split(logits, 2, axis=1)

            # With TF stuff
            # nlaw = tf.compat.v1.distributions.Normal(mean, logstd)
            # return tf.squeeze(nlaw.sample(1), axis=0)

            # Problem so far: logits goes to nan, nan at first update..
            print(f"----------{logits}---------")

            return mean + tf.exp(logstd) * tf.random.normal(tf.shape(mean))
        else:
            return tf.convert_to_tensor(
                tf.constant([[0.0 for _ in range(tf.shape(mean)[-1])]]),
                dtype=tf.float32
            )
