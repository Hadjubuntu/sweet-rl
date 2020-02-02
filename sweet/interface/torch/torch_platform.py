from sweet.interface.ml_platform import MLPlatform
from sweet.interface.torch.default_models import str_to_model
from sweet.interface.torch.torch_custom_losses import loss_actor_critic

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions.categorical as cat
import torch.nn.functional as F


class TorchPlatform(MLPlatform):
    def __init__(self, model, loss, optimizer, lr, state_shape, action_size):
        """
        Initialize Torch platform
        """
        super().__init__('torch')
        self.model = self._build_model(model, state_shape, action_size)
        self.loss = self._build_loss(loss)
        self.optimizer = self._build_optimizer(optimizer, lr)
        

    def sample(self, logits):
        """
        Sample distribution
        """
        logits_tensor = torch.tensor(logits)
        logits_tensor_soft = F.softmax(logits_tensor, dim=-1)
        m = cat.Categorical(logits_tensor_soft)

        return m.sample()

    def fast_predict(self, x):
        """
        Model prediction against observation x
        """
        if isinstance(x, list):
            tensors = []
            for element in x:
                tensors.append(torch.tensor(element))

            x = tensors
        else:
            x = torch.tensor(x)

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
        s
        """
        if isinstance(x, list):
            res = []
            for element in x:
                res.append(torch.tensor(element))
            x = res
        else:
            x = torch.tensor(x)
        y_pred = self.model(x)
        y = torch.tensor(y).float()

        loss = self.loss(y_pred, y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

    def _build_loss(self, loss: str):
        loss_out = None

        if loss == 'mean_squared_error' or loss == 'mse':
            loss_out = nn.MSELoss()
        elif loss == 'actor_categorical_crossentropy':
            loss_out = loss_actor_critic()
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

    def _build_model(self, model='dense', state_shape=None, action_shape=None):
        """
        """
        model_output = model

        if isinstance(model, str):
            model_output = str_to_model(
                model,
                input_shape=state_shape,
                output_shape=action_shape
            )

        return model_output

    def save(self, target_path):
        if not target_path.endswith('.pth'):
            target_path = f"{target_path}.pth"

        # TODO fix this
        checkpoint = {'model': self.model,
                      'state_dict': self.model.state_dict(),
                      'optimizer': self.optimizer.state_dict()}

        torch.save(checkpoint, target_path)