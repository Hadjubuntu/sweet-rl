
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense, Conv2D, Input, LSTM, Embedding, Dropout, Activation, Flatten
)


def dense(input_shape, dist, name=None):
    """
    Build a simple Dense model

    Parameters
    ----------
        input_shape: shape
            Input shape
        output_shape: int
            Number of actions (Discrete only so far)
    Returns
    -------
        model: Model
            Keras tf model
    """
    # Create inputs
    inputs = Input(shape=input_shape)
    x = Flatten()(inputs)

    # Create one dense layer and one layer for output
    x = Dense(256, activation='tanh')(x)
    x = Dense(256, activation='tanh')(x)
    
    predictions = dist.pd_from_latent(x)

    # Finally build model
    model = Model(inputs=inputs, outputs=predictions, name=name)
    model.summary()

    return model


def pi_actor_critic(input_shape, dist):
    # Create inputs
    inputs = Input(shape=input_shape)
    advs = Input(shape=1)
    x = Flatten()(inputs)

    # Create one dense layer and one layer for output
    nb_features = 256
    xa = Dense(nb_features, activation='relu')(x)
    xa = Dense(nb_features, activation='relu')(xa)

    xv = Dense(nb_features, activation='tanh')(x)
    xv = Dense(nb_features, activation='tanh')(xv)

    logits = dist.pd_from_latent(xa)
    value = Dense(1, activation='linear')(xv)

    # Finally build model
    model = Model(
        inputs=[inputs, advs],
        outputs=[logits, advs, value],
        name="pi"
    )
    model.summary()

    return model


def str_to_model(model_str: str, input_shape, dist, n_layers=1):
    """
    Build model from string:

    'dense': Dense neural network
    'conv': Convolutionnal neural network
    """
    if model_str == 'dense':
        return dense(input_shape, dist)
    elif model_str == 'pi_actor_critic':
        return pi_actor_critic(input_shape, dist)
    else:
        raise NotImplementedError(f"Unknow model: {model_str}")