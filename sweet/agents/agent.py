from abc import ABC, abstractmethod
from sweet.common.schedule import Schedule
import numpy as np

class Agent(ABC):
    def __init__(self, lr):
        """
        Abstract class to describe any RL agent
        """
        self.lr = lr

    @abstractmethod
    def act(self, obs):
        """
        Determines action regarding current observation
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



    def entropy(self, labels, base=None):
        value, counts = np.unique(labels, return_counts=True)
        norm_counts = counts / counts.sum()
        base = e if base is None else base
        
        return -(norm_counts * np.log(norm_counts)/np.log(base)).sum()
    