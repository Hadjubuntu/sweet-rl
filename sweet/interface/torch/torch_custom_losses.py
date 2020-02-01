import numpy as np

import torch
import torch.nn as nn


def loss_actor_critic(_entropy_coeff=0.0001):
    """
    Loss for actor-part of actor-critic algorithm: policy loss + entropy
    """
    cross_entrop = nn.CrossEntropyLoss()
    entropy_coeff = _entropy_coeff

    def pi_loss(y_pred_and_advs, y_true): # TODO why its different order than TF ???
        y_pred, advs = y_pred_and_advs[0], y_pred_and_advs[1]
        # First, one-hot encoding of true value y_true
        y_true = y_true.unsqueeze(-1)
        # No-need ? y_true = tf.one_hot(, self.action_size)
        print(f"y_true={y_true.size()}")
        print(f"y_pred={y_pred.size()}")
        print(f"advs={advs.size()}")
        # TIFX me: need one hot

        # Execute categorical crossentropy
        neglogp = cross_entrop(
            y_true,  # True actions chosen
            y_pred,  # Logits from model
            # sample_weight=advs
        )
        policy_loss = tf.reduce_mean(advs * neglogp)

        entropy_loss = cross_entrop(
            y_pred, y_pred
        )

        print(f"policy_loss = {policy_loss}")
        return policy_loss - entropy_coeff * entropy_loss

    # Return a function
    return pi_loss