
from sweet.agents.agent import Agent
from sweet.models.default_models import dense
from sweet.common.schedule import ConstantSchedule, LinearSchedule

from collections import deque
from keras.models import Model, Sequential
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Flatten
from keras.optimizers import Adam
import numpy as np
import random
import logging

class DqnAgent(Agent):
    """
    Simple implementation of DQN algorithm with Keras

    Inspired by:
    paper : https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
    example : https://keon.io/deep-q-learning/

    Parameters
    ----------
        state_shape: shape
            Observation state shape
        action_size: int
            Number of actions (Discrete only so far)
        model: Model or str
            Neural network model or string representing NN (dense, cnn)
        lr: float
            Learning rate
        gamma: float
            Discount factor
        epsilon: float
            Exploration factor
    """
    def __init__(self,
                state_shape,
                action_size,
                model='dense',
                lr=ConstantSchedule(0.01),
                gamma=0.99,
                epsilon=0.9):

        super().__init__(lr)

        self.state_shape = state_shape
        self.action_size = action_size

        # Hyperparameters
        self.gamma = gamma

        self.replay_buffer = deque(maxlen=500)
        self.eps = epsilon
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.5
        self.nupdate = 0

        self.model = self._build_model(model)


    def memorize(self, st, at, rt, s_t1, done, q_prediction):
        self.replay_buffer.append((st, at, rt, s_t1, done, q_prediction))
   

    def act(self, obs):
        """
        Select action
        """        
        a = None
        # Reshape obs
        obs = np.expand_dims(obs, axis=0)        
        act_values = self.model.predict(obs)

        if np.random.rand() <= self.eps:
            a = np.random.randint(low=0, high=self.action_size)
        else:
            a = np.argmax(act_values[0])

        return a, act_values

    def _build_model(self, model):
        if isinstance(model, str):
            model = dense(input_shape=self.state_shape, output_shape=self.action_size)

        model.compile(
            loss='mse',
            optimizer=Adam(lr=self._lr())
            )

        return model

    def update(self, batch_size=32):        
        if len(self.replay_buffer) > batch_size:
            self._update(batch_size)
            self.nupdate += 1

    def _update(self, batch_size):
        minibatch = random.sample(self.replay_buffer, batch_size)
        for state, action, reward, next_state, done, q_prediction in minibatch:
            target = reward
            
            if not done:
                next_state = np.expand_dims(next_state, axis=0)
                target = reward + self.gamma * \
                        np.amax(self.model.predict(next_state)[0])
            

            state =  np.expand_dims(state, axis=0)
            target_f = self.model.predict(state)
            target_f[0][action] = target
            
            history = self.model.fit(state, target_f, epochs=1, verbose=0)

            if self.nupdate % 100 == 0:
                logging.info('mse={} / eps={}'.format(history.history['loss'], self.eps))

        if self.eps > self.epsilon_min:
            self.eps *= self.epsilon_decay



        