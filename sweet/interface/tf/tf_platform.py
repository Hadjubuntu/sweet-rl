from sweet.interface.ml_platform import MLPlatform
from sweet.interface.tf.default_models import str_to_model
from sweet.interface.tf.tf_custom_losses import loss_actor_critic
from sweet.interface.tf.tf_distributions import (
    TFDistribution, TFCategoricalDist, TFDiagGaussianDist
)

import tensorflow as tf
import tensorflow.keras.losses as kl
import tensorflow.keras.optimizers as ko
from tensorflow.keras.metrics import Mean

import gym


class TFPlatform(MLPlatform):
    def __init__(
        self,
        model,
        loss,
        optimizer,
        lr,
        state_shape,
        action_space,
        **kwargs
    ):
        """
        Initialize tensorflow platform
        """
        super().__init__('tensorflow')

        self.action_space = action_space

        # Depending on action space, build distribution
        self.distribution = self._build_distribution(self.action_space)

        # Construit model, loss, optimizer
        self.loss = self._build_loss(loss, self.distribution, **kwargs)
        self.optimizer = self._build_optimizer(optimizer, lr)
        self.model = self._build_model(model, state_shape, self.distribution)

        # Loss evaluator
        self.eval_loss = Mean('loss')

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

    def _build_loss(self, loss: str, dist: TFDistribution, **kwargs):
        loss_out = None

        if loss == 'mean_squared_error' or loss == 'mse':
            loss_out = kl.MeanSquaredError()
        elif loss == 'actor_categorical_crossentropy':
            coeff_vf = kwargs.get('coeff_vf', 0.5)
            coeff_entropy = kwargs.get('coeff_entropy', 0.001)

            loss_out = loss_actor_critic(
                dist,
                _coeff_vf=coeff_vf,
                _coeff_entropy=coeff_entropy
            )
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

    def _build_model(self, model='dense', state_shape=None, dist=None):
        """
        Build model from TF-Keras model or string
        """
        model_output = model

        if isinstance(model, str):
            model_output = str_to_model(
                model,
                input_shape=state_shape,
                dist=dist
            )

        model_output.compile(loss=self.loss, optimizer=self.optimizer)

        return model_output

    def _build_distribution(self, action_space):
        """
        Build distribution from action space
        
        Parameters
        ----------
            action_space: gym.spaces.Space
                Action space
        Returns
        -------
            distribution: sweet.interface.tf.tf_distributions.TFDistribution
                Output distribution
        """
        if isinstance(action_space, gym.spaces.Discrete):
            return TFCategoricalDist(n_cat=action_space.n)
        elif isinstance(action_space, gym.spaces.Box):
            assert len(action_space.shape) == 1, "Box dim > 1 action space"
            return TFDiagGaussianDist(size=action_space.shape[0])
        else:
            raise NotImplementedError(
                (
                    f"Distribution for {type(action_space)}"
                    "action space not implemented"
                )
            )

    def save(self, target_path):
        if not target_path.endswith('.h5'):
            target_path = f"{target_path}.h5"

        self.model.save(target_path)
