
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam

from sweet.models.default_models import dense

class A2CActor():
    def __init__(self, lr, input_shape, output_shape):
        self.lr = lr
        self.model = dense(input_shape, output_shape)
        
        # Initialize model
        self.model.compile(
            loss='categorical_crossentropy',
            optimizer=Adam(lr=lr)
            )

    def act(self, obs):
        """
        Compute current policy action
        """
        return self.model.predict(obs)

    def update(self):
        pass