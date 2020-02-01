from sweet.interface.ml_platform import MLPlatform

import tensorflow as tf
import tensorflow.keras.losses as kl
import tensorflow.keras.optimizers as ko
from tensorflow.keras.metrics import Mean
from sweet.models.default_models import dense


class TFPlatform(MLPlatform):
    def __init__(self, model, loss, optimizer, lr, state_shape, action_size):
        """
        Initialize tensorflow platform
        """
        super().__init__('tensorflow')
        self.loss = self._build_loss(loss)
        self.optimizer = self._build_optimizer(optimizer, lr)

        self.model = self._build_model(model, state_shape, action_size)
        self.eval_loss = Mean('loss')

    def sample(self, logits):
        """
        Sample distribution
        """
        return tf.squeeze(tf.random.categorical(logits, 1), axis=-1)

    @tf.function
    def _graph_predict(self, x):
        return self.model(x)

    def fast_predict(self, x):
        """
        Model prediction against observation x
        """
        # [TF 2.0 error: we can't use numpy func in graph mode
        # (eg. with tf.function)] @tf.function, this is why we call a
        # sub-function
        res = self._graph_predict(x)

        # Convert from tensor to numpy array
        if isinstance(res, list):
            for idx, element in enumerate(res):
                res[idx] = element.numpy()
        else:
            res = res.numpy()

        return res

    @tf.function
    def fast_apply_gradients(self, x, y):
        """
        This is a TensorFlow function, run once for each epoch for the
        whole input. We move forward first, then calculate gradients
        with Gradient Tape to move backwards.
        """
        with tf.GradientTape() as tape:
            predictions = self.model(x)
            loss = self.loss(y, predictions)

        trainable_vars = self.model.trainable_weights

        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        return self.eval_loss(loss)

    def _build_loss(self, loss: str):
        loss_out = None

        if loss == 'mean_squared_error':
            loss_out = kl.MeanSquaredError()
        else:
            raise NotImplementedError(f'Unknow loss in TF-platform: {loss}')

        return loss_out

    def _build_optimizer(self, optimizer: str, lr: float):
        optimizer_out = None

        if optimizer == 'adam':
            optimizer_out = ko.Adam(lr)
        else:
            raise NotImplementedError(
                f'Unknow optimizer in TF-platform: {optimizer}')

        return optimizer_out

    def _build_model(self, model='dense', state_shape=None, action_shape=None):
        """
        Build model from TF-Keras model or string
        """
        if isinstance(model, str):
            model = dense(input_shape=state_shape, output_shape=action_shape)

        model.compile(loss=self.loss, optimizer=self.optimizer)

        return model

    def save(self, target_path):
        self.model.save(target_path)