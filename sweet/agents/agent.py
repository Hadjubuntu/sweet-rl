from abc import ABC, abstractmethod
import numpy as np

class Agent(ABC):
    @abstractmethod
    def act(self, obs):
        pass

    def entropy(labels, base=None):
        value, counts = np.unique(labels, return_counts=True)
        norm_counts = counts / counts.sum()
        base = e if base is None else base
        
        return -(norm_counts * np.log(norm_counts)/np.log(base)).sum()
    