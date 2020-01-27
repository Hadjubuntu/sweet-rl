import tensorflow.keras.backend as K
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam, RMSprop

from sweet.models.default_models import dense



class A2CActor():
    def __init__(self, lr, input_shape, output_shape):
        self.lr = lr
        self.model = dense(input_shape, output_shape, output_activation='softmax')
        

        # Initialize model
        self.model.compile(
            loss='categorical_crossentropy',#self.loss_func()
            optimizer=Adam(lr=lr)
        )

        # Define custom loss
    def loss_func(self):

        # Create a loss function that adds the MSE loss to the mean of all squared activations of a specific layer
        def loss(y_true,y_pred):
            return K.mean(K.square(y_pred - y_true), axis=-1)
    
        # Return a function
        return loss

    def act(self, obs):
        """
        Compute current policy action
        """
        return self.model.predict(obs)

    def update(self, obs, actions, advs):
        # TODO
        # Create own loss
        """
        """
        history = self.model.fit(obs, advs, epochs=1, verbose=0)