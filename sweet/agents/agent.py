from abc import ABC, abstractmethod
from sweet.common.schedule import Schedule
import numpy as np

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam

from sweet.models.default_models import dense

class Agent(ABC):
    """
    Generic class for RL algorithm
    """    
    def __init__(self, lr, model, state_shape, action_size):
        """
        Generic initialization of RL algorithm
        """
        # Set common hyperparameters
        self.lr = lr

        # Initialize model
        self.model = self._build_model(model, state_shape, action_size)

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
        loss = None

        if isinstance(model, str):
            model = dense(input_shape=state_shape, output_shape=action_shape)
            loss = 'mse'

        model.compile(
            loss=loss,
            optimizer=Adam(lr=self._lr())
            )

        return model

    def _entropy(self, labels, base=None):
        """
        Compute entry of input labels
        """
        value, counts = np.unique(labels, return_counts=True)
        norm_counts = counts / counts.sum()
        base = e if base is None else base
        
        return -(norm_counts * np.log(norm_counts)/np.log(base)).sum()

    