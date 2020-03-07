from abc import ABC, abstractmethod


class TorchDistribution:
    """
    Generic class for distribution
    """
    def __init__(self):
        pass


class TorchCategoricalDist(TorchDistribution):
    def __init__(self):
        super().__init__()


class TorchDiagGaussianDist(TorchDistribution):
    def __init__(self):
        super().__init__()
