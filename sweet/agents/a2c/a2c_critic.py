
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam

from sweet.models.default_models import dense

class A2CCritic():
    def __init__(self, lr, input_shape):
        self.lr = lr
        self.model = dense(input_shape, output_shape=1)
        
        # Initialize model
        self.model.compile(
            loss='mse',
            optimizer=Adam(lr=lr)
            )

    def predict(self, obs):
        return self.model.predict(obs)

    def update(self):
        pass