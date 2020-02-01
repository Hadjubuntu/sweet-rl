from sweet.interface.ml_platform import MLPlatform
from sweet.interface.torch.default_models import dense

import torch
import torch.nn as nn
import torch.optim as optim


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
        raise NotImplementedError

    def fast_predict(self, x):
        """
        Model prediction against observation x
        """
        x = torch.tensor(x)
        output = self.model(x).detach().numpy()
        return output

    def fast_apply_gradients(self, x, y):
        """
        s
        """
        x = torch.tensor(x)
        y_pred = self.model(x)
        y = torch.tensor(y)

        loss = self.loss(y_pred, y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

    def _build_loss(self, loss: str):
        loss_out = None

        if loss == 'mean_squared_error' or loss == 'mse':
            loss_out = nn.MSELoss()
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
        if isinstance(model, str):
            model = dense(input_shape=state_shape, output_shape=action_shape)

        return model

    def save(self, target_path):
        if not target_path.endswith('.pth'):
            target_path = f"{target_path}.pth"

        # TODO fix this
        checkpoint = {'model': self.model,
                      'state_dict': self.model.state_dict(),
                      'optimizer': self.optimizer.state_dict()}

        torch.save(checkpoint, target_path)