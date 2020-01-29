"""
Since TF2.0 seems ways slower than TF1.4, a little code to compare
"""

import numpy as np
import time

# Supposing TF2.0 is installed
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from sweet.models.default_models import dense
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Flatten


@tf.function
def tf2_train_step(model, loss, optimizer, x,y):
    with tf.GradientTape() as tape:
        predictions = model(x)
        loss = loss(y, predictions)

    trainable_vars = model.trainable_weights

    gradients = tape.gradient(loss, trainable_vars)
    optimizer.apply_gradients(zip(gradients, trainable_vars))

def exec_tf2():

    nupdate = 30
    batch_size = 256
    input_size = 4
    output_size = 1
    lr = 0.01
    
    """
    Model
    """
    inputs = Input(shape=(input_size,))
    x = inputs
    x = Dense(24, activation='tanh')(x)
    x = Dense(24, activation='tanh')(x)
    predictions = Dense(output_size, activation='linear')(x)
    model = Model(inputs=inputs, outputs=predictions, name="model_tf2.0")
    

    """
    Optimize
    """
    optimizer = tf.keras.optimizers.Adam(lr=lr)
    loss_obj = tf.keras.losses.MeanSquaredError()

    start = time.time()
    
    for _ in range(nupdate):
        # Generate data
        batch_x = np.random.rand(batch_size, input_size).astype(np.float64)
        batch_y = np.random.randint(low=0, high=1, size=(batch_size, output_size))
        # Gradient descent
        tf2_train_step(model, loss_obj, optimizer, batch_x, batch_y)

    dt = time.time() - start
    print(f"dt = {dt}")

if __name__ == "__main__":
    exec_tf2()