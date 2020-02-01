from abc import ABC, abstractmethod


class MLPlatform(ABC):
    """
    Generic API to Machine Learning platform
    """
    def __init__(self, name=None):
        """
        """
        self.name = name

    @abstractmethod
    def sample(self, logits):
        """
        Sample values from probability distribution
        """
        pass

    @abstractmethod
    def fast_predict(self, x):
        """
        Model prediction from observation x
        """
        pass

    @abstractmethod
    def fast_apply_gradients(self, x, y):
        """
        """
        pass

    @abstractmethod
    def save(self, target_path):
        pass
    