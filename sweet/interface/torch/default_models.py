import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TorchDense(nn.Module):

    def __init__(self, state_shape, action_size: int):
        super(TorchDense, self).__init__()
        input_size_flatten = self.num_flat_features(state_shape)

        self.h1 = nn.Linear(input_size_flatten, 128)
        self.out = nn.Linear(128, action_size)

    def forward(self, x):
        x = torch.relu(self.h1(x))
        x = torch.relu(self.out(x))
        return x

    def num_flat_features(self, x):
        return np.sum(x)


def str_to_model(model_str: str, n_layers=1):
    """
    Build model from string:

    'dense': Dense neural network
    'conv': Convolutionnal neural network
    """
    if model_str == 'dense':
        return Torch
    else:
        raise NotImplementedError(f"Unknow model: {model_str}")


def dense(input_shape, output_shape, output_activation='linear', name=None):
    model = TorchDense(input_shape, output_shape)

    return model