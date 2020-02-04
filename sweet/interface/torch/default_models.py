import numpy as np
import torch
import torch.nn as nn


class TorchDense(nn.Module):
    def __init__(self, state_shape, action_size: int):
        super(TorchDense, self).__init__()
        input_size_flatten = self.num_flat_features(state_shape)

        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
        self.h1 = nn.Linear(input_size_flatten, 16)
        self.h2 = nn.Linear(16, 16)
        self.out = nn.Linear(16, action_size)

    def forward(self, x):
        x = self.flatten(x)
        x = torch.tanh(self.h1(x))
        x = torch.tanh(self.h2(x))
        x = self.out(x)
        return x

    def num_flat_features(self, x):
        return np.prod(x)


class TorchPiActor(nn.Module):
    def __init__(self, state_shape, action_size: int):
        super(TorchPiActor, self).__init__()
        input_size_flatten = self.num_flat_features(state_shape)

        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
        self.h1 = nn.Linear(input_size_flatten, 16)
        self.out = nn.Linear(16, action_size)

    def forward(self, x):
        """
        x1: observation
        x2: given advantages
        """
        x1 = x[0]
        x2 = x[1]

        x = self.flatten(x1)
        x = torch.relu(self.h1(x))
        x = self.out(x)

        return [x, x2]

    def num_flat_features(self, x):
        return np.prod(x)


def str_to_model(model_str: str, input_shape, output_shape, n_layers=1):
    """
    Build model from string:

    'dense': Dense neural network
    'conv': Convolutionnal neural network
    """
    if model_str == 'dense':
        return TorchDense(input_shape, output_shape)
    elif model_str == 'pi_actor':
        return TorchPiActor(input_shape, output_shape)
    else:
        raise NotImplementedError(f"Unknow model: {model_str}")
