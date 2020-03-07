"""
This module aims to describe how to implement
your own model instead of using default ones.

Of course you shall have consistency between model creation
and ML platform configuration.
"""
import gym
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense, Input, LSTM, Embedding, Dropout, Activation, Flatten
)

from sweet.agents.dqn.train import learn
from sweet.interface.tf.tf_platform import TFPlatform

from sweet.interface.tf.tf_distributions import TFCategoricalDist


def custom_model(env_name):
    # Create env to retrieve state shape and action space
    env = gym.make(env_name)
    input_shape = env.observation_space.shape
    dist = TFCategoricalDist(n_cat=env.action_space.n)

    # Then create TF 2.0 model
    inputs = Input(shape=input_shape)
    x = Flatten()(inputs)

    x = Dense(128, activation='tanh')(x)
    x = Dense(128, activation='tanh')(x)

    # Build distribution for discrete action space
    predictions = dist.pd_from_latent(x)

    # Finally build model
    model = Model(inputs=inputs, outputs=predictions, name='custom_model')
    model.summary()

    return model


def experiment_custom_model(env_name):
    # Create custom model
    model = custom_model(env_name)

    # Train model
    learn(
        ml_platform=TFPlatform,
        env_name=env_name,
        model=model,
        total_timesteps=1e5,
        lr=0.0003
    )

if __name__ == "__main__":
    experiment_custom_model(env_name='CartPole-v0')
