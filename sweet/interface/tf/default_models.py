
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense, Input, LSTM, Embedding, Dropout, Activation, Flatten
)


def dense(input_shape, output_shape, output_activation='linear', name=None):
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
    x = Dense(8, activation='tanh')(x)
    x = Dense(8, activation='tanh')(x)
    predictions = Dense(output_shape, activation='linear')(x)

    # Finally build model
    model = Model(inputs=inputs, outputs=predictions, name=name)
    model.summary()

    return model


def pi_actor(input_shape, output_shape):
    # Create inputs
    inputs = Input(shape=input_shape)
    advs = Input(shape=1)
    x = Flatten()(inputs)

    # Create one dense layer and one layer for output
    x = Dense(8, activation='relu')(x)
    x = Flatten()(x)
    logits = Dense(output_shape)(x)

    # Finally build model
    model = Model(inputs=[inputs, advs], outputs=[logits, advs], name="pi")
    model.summary()

    return model


def str_to_model(model_str: str, input_shape, output_shape, n_layers=1):
    """
    Build model from string:

    'dense': Dense neural network
    'conv': Convolutionnal neural network
    """
    if model_str == 'dense':
        return dense(input_shape, output_shape)
    elif model_str == 'pi_actor':
        return pi_actor(input_shape, output_shape)
    else:
        raise NotImplementedError(f"Unknow model: {model_str}")