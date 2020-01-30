
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam

from sweet.models.default_models import dense
from sweet.agents.agent import Agent
from tensorflow.keras.losses import MeanSquaredError


from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Flatten

def custom_dense(input_shape, output_shape, output_activation='linear', name=None):
    # Create inputs
    inputs = Input(shape=input_shape)
    x = inputs

    # Create one dense layer and one layer for output
    x = Dense(128, activation='relu')(x)
    predictions = Dense(output_shape, activation=output_activation)(x)

    # Finally build model
    model = Model(inputs=inputs, outputs=predictions, name=name)
    model.summary()

    return model


class A2CCritic(Agent):
    def __init__(self, lr, input_shape):
        """
        Critic part of A2C is an estimator of the value V(s).
        It is used to compute the function advantage: Adv(s,a)=Q(s,a)-V(s)
        during policy optimization.

        Parameters
        ----------
            lr: float or sweet.common.schedule.Schedule
                Learning rate
            input_shape: shape
                Observation state shape
        """
        super().__init__(
            lr,
            None,
            input_shape,
            1,
            optimizer=Adam,
            loss=MeanSquaredError
        )

        # FIXME override model
        self.model = custom_dense(input_shape, output_shape=1, name='A2C_critic')

        # Initialize model
        self.model.compile(
            loss=self.loss,
            optimizer=self.optimizer
        )

    def predict(self, obs):
        """
        Predict state value V(s)
        """
        V_s = self.tf2_fast_predict(obs)
        return V_s

    def update(self, obs, values):
        """
        Update critic network
        """
        loss = self.tf2_fast_apply_gradients(obs, values)
        return loss

    def act(self, obs):
        pass
