import numpy as np
import os
from abc import ABC, abstractmethod
from pathlib import Path

from sweet.common.schedule import Schedule
from sweet.interface.ml_platform import MLPlatform


class Agent(ABC):
    """
    Generic class for RL algorithm
    """
    def __init__(
            self,
            ml_platform: MLPlatform,
            lr,
            model,
            state_shape,
            action_size,
            optimizer: str = 'adam',
            loss: str = 'mean_squared_error'):
        """
        Generic initialization of RL algorithm
        """
        # Input/output shapes
        self.state_shape = state_shape
        self.action_size = action_size

        # Set common hyperparameters
        self.lr = lr

        # Initialize model/optimizer and loss wrt to Machine Learning platform
        # (so far, IF to TF2 or Torch are implemented)
        self.ml_platform = ml_platform(
            model,
            loss,
            optimizer,
            self.lr,
            state_shape,
            action_size
        )

    def sample(self, logits):
        """
        Sample distribution
        """
        return self.ml_platform.sample(logits)

    def encode(self, var):
        """
        From raw observation to encoded obs for neural network application
        """
        var = np.expand_dims(var, axis=0)
        return var.astype(np.float32)

    def save_model(self, target_path: Path):
        """
        Save model to file (HDF5 for tensorflow, Pth for Torch)
        """
        os.makedirs(target_path.parent, exist_ok=True)
        target_path = str(target_path)
        self.ml_platform.save(target_path)

    def fast_predict(self, x):
        """
        Model prediction against observation x
        """
        return self.ml_platform.fast_predict(x)

    def fast_apply_gradients(self, x, y):
        """
        Update model wrt (x, y) data batch
        """
        loss = self.ml_platform.fast_apply_gradients(x, y)
        
        return loss

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
