
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam

from sweet.models.default_models import dense
from sweet.agents.agent import Agent
from tensorflow.keras.losses import MeanSquaredError


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
            'dense',
            input_shape,
            1,
            optimizer=Adam,
            loss=MeanSquaredError
        )

        # FIXME override model
        self.model = dense(input_shape, output_shape=1, name='A2C_critic')

        # Initialize model
        self.model.compile(
            loss=self.loss,
            optimizer=self.optimizer
        )

    def predict(self, obs):
        """
        Predict state value V(s)
        """
        return self.tf2_fast_predict(obs)

    def update(self, obs, values):
        """
        Update critic network
        """
        loss = self.tf2_fast_apply_gradients(obs, values)
        return loss

    def act(self, obs):
        pass
