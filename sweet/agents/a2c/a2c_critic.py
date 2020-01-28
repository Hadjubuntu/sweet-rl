
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam

from sweet.models.default_models import dense

class A2CCritic():
    def __init__(self, lr, input_shape):
        """
        Initialize critic network of A2C

        Parameters
        ----------
            lr: float or sweet.common.schedule.Schedule
                Learning rate
            input_shape: shape
                Observation state shape
        """
        self.lr = lr
        self.model = dense(input_shape, output_shape=1, name='A2C_critic')
        
        # Initialize model
        self.model.compile(
            loss='mse',
            optimizer=Adam(lr=lr)
            )

    def predict(self, obs):
        """
        Predict state value V(s)
        """
        return self.model.predict(obs)

    def update(self, obs, values):
        """
        Update critic network
        """
        history = self.model.train_on_batch(obs, values)
        return history