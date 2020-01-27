
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Flatten

import logging

def str_to_model(model_str: str):
    if model_str == 'dense':
        return None
    else:
        logging.error(f"Unknow model: {model_str}")


def dense(input_shape, output_shape):
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

    # Create one dense layer and one layer for output
    x = Dense(64, activation='relu')(inputs)
    predictions = Dense(output_shape, activation='linear')(x)

    # Finally build model
    model = Model(inputs=inputs, outputs=predictions)
    model.summary()

    return model