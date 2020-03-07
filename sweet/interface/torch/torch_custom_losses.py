
import torch.nn as nn
from torch.distributions import Categorical


def loss_actor_critic(_dist, _coeff_vf=0.5, _coeff_entropy=0.001):
    """
    Loss for actor-part of actor-critic algorithm: policy loss + entropy
    """
    dist = _dist

    # cross_entrop = nn.CrossEntropyLoss()
    mse = nn.MSELoss()
    coeff_vf = _coeff_vf
    coeff_entropy = _coeff_entropy

    # TODO why its different order than TF ???
    def pi_loss(m_out, y_true):
        y_pred, advs, vf = m_out[0], m_out[1], m_out[2]

        y_true_action = y_true[0].long()
        vf_true = y_true[1]

        # Execute categorical crossentropy
        neglogp = dist.neglogp(
            y_pred,  # Logits from model
            y_true_action,  # True actions chosen
            # sample_weight=advs
        )
        policy_loss = (advs * neglogp).mean()

        entropy_loss = dist.entropy(y_pred).mean()

        loss_vf = mse(vf, vf_true)

        return policy_loss - coeff_entropy * entropy_loss + coeff_vf * loss_vf

    # Return a function
    return pi_loss
