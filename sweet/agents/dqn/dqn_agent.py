
from collections import deque
from keras.models import Model, Sequential
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Flatten
from keras.optimizers import Adam
import numpy as np
import random

class DqnAgent():
    """
    Simple implementation of DQN algorithm with Keras

    Inspired by:
    paper : https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
    example : https://keon.io/deep-q-learning/
    """
    def __init__(self, 
                state_shape, 
                action_size, 
                lr=0.001,
                timesteps=1e4):
        self.timesteps = timesteps
        self.lr = lr
        self.state_shape = state_shape
        self.action_size = action_size

        self.gamma = 0.9    # discount rate

        self.replay_buffer = deque(maxlen=10000)
        self.eps = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.99
        self.nupdate = 0

        self.model = self._build_model()


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

    def _build_model(self):
        # This returns a tensor
        inputs = Input(shape=self.state_shape)

        # a layer instance is callable on a tensor, and returns a tensor
        x = Dense(32, activation='relu')(inputs)
        # x = Dense(32, activation='relu')(x)
        predictions = Dense(self.action_size, activation='linear')(x)

        model = Model(inputs=inputs, outputs=predictions)
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.lr))

        model.summary()

        return model

    def update(self, batch_size=64):
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

            mse = (np.square(target_f - q_prediction)).mean(axis=None)
            print("Diff target_f = {} // pred = {} // mse = {}".format(target_f, q_prediction, mse))
            
            history = self.model.fit(state, target_f, epochs=1, verbose=0)

            if self.nupdate % 100 == 0:
                print('mse={} / eps={}'.format(history.history['loss'], self.eps))

        if self.eps > self.epsilon_min:
            self.eps *= self.epsilon_decay



        