import tensorflow.keras.backend as K
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam, RMSprop

from sweet.models.default_models import dense



class A2CActor():
    def __init__(self, lr, input_shape, output_shape):
        """
        Initialize critic network of A2C
        
        Parameters
        ----------
            lr: float or sweet.common.schedule.Schedule
                Learning rate
            input_shape: shape
                Observation state shape
            output_shape: int
                Number of actions (Discrete only so far)
        """
        self.lr = lr
        self.model = dense(input_shape, output_shape, output_activation='softmax', name='A2C_actor')
        

        # Initialize model
        self.model.compile(
            loss='categorical_crossentropy',#self.loss_func()
            optimizer=Adam(lr=lr)
        )

    def loss_func(self):
        """
        Custom loss func to add entropy
        """
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
        """
        Update actor network
        """
        history = self.model.train_on_batch(obs, advs)
        return history