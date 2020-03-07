from abc import ABC, abstractmethod
import torch
import torch.nn.functional as F
import torch.distributions.categorical as cat
import torch.nn as nn
from torch.distributions import Categorical


class TorchDistribution:
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


class TorchCategoricalDist(TorchDistribution):
    def __init__(self, n_cat):
        super().__init__()
        self.n_cat = n_cat
        self.cross_entrop = nn.CrossEntropyLoss()

    def sample(self, logits):
        """
        Sample distribution
        """
        logits_tensor = torch.tensor(logits)
        logits_tensor_soft = F.softmax(logits_tensor, dim=-1)
        m = cat.Categorical(logits_tensor_soft)

        return m.sample()

    def pd_from_latent(self, x, prev_size):
        return nn.Linear(prev_size, self.n_cat)(x)

    def neglogp(self, x_true, x):
        return self.cross_entrop(x_true, x)

    def entropy(self, x):
        cat = Categorical(x)
        return cat.entropy()


class TorchDiagGaussianDist(TorchDistribution):
    def __init__(self):
        super().__init__()
