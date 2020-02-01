from sweet.agents.agent import Agent
from sweet.interface.ml_platform import MLPlatform


class A2CCritic(Agent):
    def __init__(
        self,
        ml_platform: MLPlatform,
        lr,
        input_shape
    ):
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
            ml_platform=ml_platform,
            lr=lr,
            model='dense',
            state_shape=input_shape,
            action_size=1,
            optimizer='adam',
            loss='mean_squared_error'
        )

    def predict(self, obs):
        """
        Predict state value V(s)
        """
        V_s = self.fast_predict(obs)
        return V_s

    def update(self, obs, values):
        """
        Update critic network
        """
        loss = self.fast_apply_gradients(obs, values)
        return loss

    def act(self, obs):
        pass
