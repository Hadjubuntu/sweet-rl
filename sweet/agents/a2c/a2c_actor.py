import tensorflow.keras.backend as K
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.losses import CategoricalCrossentropy

from sweet.models.default_models import dense
from sweet.agents.agent import Agent


class A2CActor(Agent):
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

        super().__init__(
            lr,
            'dense',
            input_shape,
            output_shape,
            optimizer=Adam,
            loss=CategoricalCrossentropy
        )

        self.model = dense(
            input_shape,
            output_shape,
            output_activation='softmax',
            name='A2C_actor'
        )

        # Initialize model
        self.model.compile(
            loss=self.loss,
            optimizer=self.optimizer
        )

    def loss_func(self):
        """
        Custom loss func to add entropy
        """
        # Create a loss function that adds the MSE loss to the mean of all
        # squared activations of a specific layer
        def loss(y_true, y_pred):
            return K.mean(K.square(y_pred - y_true), axis=-1)

        # Return a function
        return loss

    def act(self, obs):
        """
        Compute current policy action
        """
        return self.tf2_fast_predict(obs)

    def update(self, obs, actions, advs):
        """
        Update actor network
        """
        loss = 0.0  # FIXME

        self.tf2_fast_apply_gradients(obs, advs)
        return loss
