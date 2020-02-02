import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Categorical


# Memo:
# y_true_hot = torch.zeros(y_true.size()[0], 2)
# y_true_hot[torch.arange(y_true.size()[0]), y_true] = 1
# y_true = y_true_hot

def loss_actor_critic(_entropy_coeff=0.0001):
    """
    Loss for actor-part of actor-critic algorithm: policy loss + entropy
    """
    cross_entrop = nn.CrossEntropyLoss()
    entropy_coeff = _entropy_coeff

    # TODO why its different order than TF ???
    def pi_loss(y_pred_and_advs, y_true):
        y_pred, advs = y_pred_and_advs[0], y_pred_and_advs[1]

        y_true = y_true.long()

        # print(f"Dimension ==========+> {y_true.size()}")
        # print(f"y_true={y_true.size()}")
        # print(f"y_pred={y_pred.size()}")
        # print(f"advs={advs.size()}")

        # Execute categorical crossentropy
        neglogp = cross_entrop(
            y_pred,  # Logits from model
            y_true,  # True actions chosen
            # sample_weight=advs
        )
        policy_loss = (advs * neglogp).mean()

        cat = Categorical(y_pred)
        entropy_loss = cat.entropy().mean()

        return policy_loss - entropy_coeff * entropy_loss

    # Return a function
    return pi_loss
