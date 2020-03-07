from sweet.interface.ml_platform import MLPlatform
from sweet.interface.torch.default_models import str_to_model
from sweet.interface.torch.torch_custom_losses import loss_actor_critic
from sweet.interface.torch.torch_distributions import (
    TorchDistribution, TorchCategoricalDist, TorchDiagGaussianDist
)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions.categorical as cat
import torch.nn.functional as F

import gym


class TorchPlatform(MLPlatform):
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
        Initialize Torch platform
        """
        super().__init__('torch')

        self.action_space = action_space

        # Temporary
        # Only dicrete action so far
        action_size = action_space.n  # Only dicrete action so far


        # Depending on action space, build distribution
        self.distribution = self._build_distribution(self.action_space)

        # Construit model, loss, optimizer
        self.model = self._build_model(model, state_shape, self.distribution)
        self.loss = self._build_loss(loss, **kwargs)
        self.optimizer = self._build_optimizer(optimizer, lr)

   
    def fast_predict(self, x):
        """
        Model prediction against observation x
        """
        x = self._to_tensor(x)

        raw_output = self.model(x)

        # Convert from tensor to numpy array
        if isinstance(raw_output, list):
            output = raw_output
            for idx, element in enumerate(raw_output):
                output[idx] = element.detach().numpy()

        else:
            output = raw_output.detach().numpy()

        return output

    def fast_apply_gradients(self, x, y):
        """
        Apply gradient
        """
        x = self._to_tensor(x)
        y = self._to_tensor(y)
            
        y_pred = self.model(x)
        loss = self.loss(y_pred, y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

    def _to_tensor(self, x):
        """
        Transform numpy array into tensor
        or list of np.array to list of tensors
        """
        if isinstance(x, list):
            res = []
            for element in x:
                res.append(torch.tensor(element).float())
            x = res
        else:
            x = torch.tensor(x).float()
        return x

    def _build_loss(self, loss: str, **kwargs):
        loss_out = None

        if loss == 'mean_squared_error' or loss == 'mse':
            loss_out = nn.MSELoss()
        elif loss == 'actor_categorical_crossentropy':
            coeff_vf = kwargs.get('coeff_vf', 0.5)
            coeff_entropy = kwargs.get('coeff_entropy', 0.001)

            loss_out = loss_actor_critic(
                _coeff_vf=coeff_vf, _coeff_entropy=coeff_entropy
            )
        else:
            raise NotImplementedError(f'Loss not implemented so far: {loss}')

        return loss_out

    def _build_optimizer(self, optimizer: str, lr: float):
        if optimizer == 'adam':
            optimizer_out = optim.Adam(self.model.parameters(), lr=lr)
        else:
            raise NotImplementedError(
                f"Optimizer not implemented so far: {optimizer}")

        return optimizer_out

    def _build_model(self, model='dense', state_shape=None, dist=None):
        """
        """
        model_output = model

        if isinstance(model, str):
            model_output = str_to_model(
                model,
                input_shape=state_shape,
                dist=dist
            )

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
            return TorchCategoricalDist(n_cat=action_space.n)
        elif isinstance(action_space, gym.spaces.Box):
            raise NotImplementedError(
                "Distribution for Box action space not implemented")
        else:
            raise NotImplementedError((
                f"Distribution for {type(action_space)}"
                "action space not implemented"
            ))   

    def save(self, target_path):
        if not target_path.endswith('.pth'):
            target_path = f"{target_path}.pth"

        checkpoint = {'model': self.model,
                      'state_dict': self.model.state_dict(),
                      'optimizer': self.optimizer.state_dict()}

        torch.save(checkpoint, target_path)
