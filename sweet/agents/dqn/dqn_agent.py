
from sweet.agents.agent import Agent
from sweet.common.schedule import ConstantSchedule, LinearSchedule

import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam
from sweet.models.default_models import dense
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Flatten

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
                gamma: float=0.95,
                epsilon: float=1.0,
                epsilon_min: float=0.01,
                epsilon_decay: float=0.99,
                replay_buffer: int=1000):
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


    def memorize(self, batch_data):
        """
        Store pair of state, action, reward, .. data into a buffer
        """
        for st, st_1, rt, at, done, q_prediction in batch_data:
            self.replay_buffer.append((st, st_1, rt, at, done, q_prediction))
   

    def act(self, obs):
        """
        Select action regarding exploration factor.
        """        
        a = None
        # Reshape obs
        obs = np.expand_dims(obs, axis=0)
        q_values = self.model.predict(obs)

        if np.random.rand() <= self.eps:
            a = np.random.randint(low=0, high=self.action_size)
        else:
            a = np.argmax(q_values[0])

        return a, q_values

    def decay_exploration(self, ntimes):
        for _ in range(ntimes):
            if self.eps > self.epsilon_min:
                self.eps *= self.epsilon_decay

    def loss_func(self, layer_action):
        # Create a loss function that adds the MSE loss to the mean of all squared activations of a specific layer
        def loss(y_true,y_pred):
            return K.mean(K.square(y_pred - y_true), axis=-1)
    
        # Return a function
        return loss
    
    def update(self, batch_size=16):     
        if len(self.replay_buffer) >= batch_size:
            self._update(batch_size)
            self.nupdate += 1

    def _update(self, batch_size):
        # Sample minibatch from replay buffer
        minibatch = random.sample(self.replay_buffer, batch_size)        
        x, y = [], []

        # Fit model for each minibatch data
        for batch_data in minibatch:
            state, next_state, reward, action, done, q_pred_at_state_t = batch_data
            target = reward
            
            if not done:
                next_state = np.expand_dims(next_state, axis=0)

                target = reward + self.gamma * \
                        np.amax(self.model.predict(next_state)[0])            

            state =  np.expand_dims(state, axis=0)
            target_f = self.model.predict(state)
            target_f[0][action] = target

            # TODO: optim: use mask to apply loss on action selected  
            
            # Execute gradient descent
            #loss = self.model.train_on_batch(state, target_f) 

            x.append(state[0])
            y.append(target_f[0])
        

        x=np.array(x)
        y=np.array(y)
        loss = self.model.train_on_batch(x, y) 

        logging.info('mse={} / eps={}'.format(loss, self.eps))