
from sweet.agents.agent import Agent
from sweet.common.schedule import ConstantSchedule, LinearSchedule

from collections import deque
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
        lr: float or sweet.common.schedule.Schedule
            Learning rate
        gamma: float
            Discount factor
        epsilon: float
            Exploration probability (choose random action over max Q-value action)
        epsilon_min: float
            Minimum probability of exploration
        epsilon_decay: float
            Decay of exploration at each update
        replay_buffer: int
            Size of the  replay buffer
    """
    def __init__(self,
                state_shape,
                action_size,
                model='dense',
                lr=ConstantSchedule(0.01),
                gamma: float=0.99,
                epsilon: float=0.9,
                epsilon_min: float=0.1,
                epsilon_decay: float=0.5,
                replay_buffer: int=5000):
        # Generic initialization
        super().__init__(lr, model, state_shape, action_size)

        # Input/output shapes
        self.state_shape = state_shape
        self.action_size = action_size

        # Hyperparameters
        self.gamma = gamma
        self.replay_buffer = deque(maxlen=replay_buffer)
        self.eps = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Statistics
        self.nupdate = 0


    def memorize(self, st, at, rt, s_t1, done, q_prediction):
        """
        Store pair of state, action, reward, .. data into a buffer
        """
        self.replay_buffer.append((st, at, rt, s_t1, done, q_prediction))
   

    def act(self, obs):
        """
        Select action regarding exploration factor.
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



        