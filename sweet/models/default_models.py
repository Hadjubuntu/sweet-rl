
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Flatten

import logging


def str_to_model(model_str: str, n_layers=1):
    """
    Build model from string:

    'dense': Dense neural network
    'conv': Convolutionnal neural network
    """
    if model_str == 'dense':
        return None
    else:
        logging.error(f"Unknow model: {model_str}")


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
    x = inputs

    # Create one dense layer and one layer for output
    x = Dense(128, activation='tanh')(x)
    x = Dense(128, activation='tanh')(x)
    predictions = Dense(output_shape, activation='linear')(x)

    # Finally build model
    model = Model(inputs=inputs, outputs=predictions, name=name)
    model.summary()

    return model