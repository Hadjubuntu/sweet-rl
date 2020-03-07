import numpy as np
import torch
import torch.nn as nn


class TorchDense(nn.Module):
    def __init__(self, state_shape, dist):
        super(TorchDense, self).__init__()        
        self.dist = dist

        input_size_flatten = self.num_flat_features(state_shape)

        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
        self.h1 = nn.Linear(input_size_flatten, 256)
        self.h2 = nn.Linear(256, 256)

    def forward(self, x):
        x = self.flatten(x)
        x = torch.tanh(self.h1(x))
        x = torch.tanh(self.h2(x))
        x = self.dist.pd_from_latent(x, prev_size=256)
        return x

    def num_flat_features(self, x):
        return np.prod(x)


# class TorchPiActorCritic(nn.Module):
#     def __init__(self, state_shape, dist):
#         super(TorchPiActorCritic, self).__init__()
#         action_size = dist.n()  # hot-fix
#         input_size_flatten = self.num_flat_features(state_shape)

#         self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
#         self.nb_features = 256

#         # Layers for action
#         self.ha1 = nn.Linear(input_size_flatten, self.nb_features)
#         self.ha2 = nn.Linear(self.nb_features, self.nb_features)
#         # Layers for value
#         self.hv1 = nn.Linear(input_size_flatten, self.nb_features)
#         self.hv2 = nn.Linear(self.nb_features, self.nb_features)

#         self.out = nn.Linear(self.nb_features, action_size)
#         self.out_value = nn.Linear(self.nb_features, 1)

#     def forward(self, x):
#         """
#         x1: observation
#         x2: given advantages
#         """
#         x1 = x[0]
#         x2 = x[1]

#         x = self.flatten(x1)

#         xa = torch.relu(self.ha1(x))
#         xa = torch.relu(self.ha2(xa))

#         xv = torch.tanh(self.hv1(x))
#         xv = torch.tanh(self.hv2(xa))

#         # Outputs
#         logits = self.out(xa)
#         value = self.out_value(xv)

#         return [logits, x2, value]

#     def num_flat_features(self, x):
#         return np.prod(x)

 

class TorchPiActorCritic(nn.Module):
    def __init__(self, state_shape, dist):
        super(TorchPiActorCritic, self).__init__()
        self.dist = dist

        input_size_flatten = self.num_flat_features(state_shape)

        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
        self.nb_features = 256

        # Layers for action
        self.ha1 = nn.Linear(input_size_flatten, self.nb_features)
        self.ha2 = nn.Linear(self.nb_features, self.nb_features)
        # Layers for value
        self.hv1 = nn.Linear(input_size_flatten, self.nb_features)
        self.hv2 = nn.Linear(self.nb_features, self.nb_features)

        self.out = self.dist.pd_from_latent(prev_size=self.nb_features)
        self.out_value = nn.Linear(self.nb_features, 1)

    def forward(self, x):
        """
        x1: observation
        x2: given advantages
        """
        x1 = x[0]
        x2 = x[1]
        x = self.flatten(x1)

        xa = torch.relu(self.ha1(x))
        xa = torch.relu(self.ha2(xa))

        xv = torch.tanh(self.hv1(x))
        xv = torch.tanh(self.hv2(xa))

        # Outputs
        logits = self.out(xa)
        value = self.out_value(xv)

        return [logits, x2, value]

    def num_flat_features(self, x):
        return np.prod(x)


def str_to_model(model_str: str, input_shape, dist, n_layers=1):
    """
    Build model from string:

    'dense': Dense neural network
    'conv': Convolutionnal neural network
    """
    if model_str == 'dense':
        return TorchDense(input_shape, dist)
    elif model_str == 'pi_actor_critic':
        return TorchPiActorCritic(input_shape, dist)
    else:
        raise NotImplementedError(f"Unknow model: {model_str}")
