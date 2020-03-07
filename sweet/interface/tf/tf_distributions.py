from abc import ABC, abstractmethod

from tensorflow.keras.layers import (
    Dense, Conv2D, Input, LSTM, Embedding, Dropout, Activation, Flatten
)
import tensorflow.keras.losses as kls

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

    def pd_from_latent(self, x_latent):
        logits = Dense(self.n_cat)(x_latent)
        return logits

    def neglogp(self, x_true, x):
        return kls.CategoricalCrossentropy(
            from_logits=True)(x_true, x)

    def n(self):
        return self.n_cat

    def entropy(self, x):
        return  kls.CategoricalCrossentropy(
            from_logits=True)(x, x)


class TFDiagGaussianDist(TFDistribution):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def pd_from_latent(self, x_latent):
        mean = Dense(self.size)(x_latent)
        std = Dense(self.size)(x_latent)
        return tf.concat([mean, std], axis=1)

    def n(self):
        return 2 * self.size