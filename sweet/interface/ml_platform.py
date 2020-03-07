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
    def fast_predict(self, x):
        """
        Model prediction from observation x
        """

    @abstractmethod
    def fast_apply_gradients(self, x, y):
        """
        """

    @abstractmethod
    def save(self, target_path):
        pass
    