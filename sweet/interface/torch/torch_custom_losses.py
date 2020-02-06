
import torch.nn as nn
from torch.distributions import Categorical


def loss_actor_critic(_coeff_vf=0.5, _entropy_coeff=0.00001):
    """
    Loss for actor-part of actor-critic algorithm: policy loss + entropy
    """
    cross_entrop = nn.CrossEntropyLoss()
    mse = nn.MSELoss()
    entropy_coeff = _entropy_coeff
    coeff_vf = _coeff_vf

    # TODO why its different order than TF ???
    def pi_loss(m_out, y_true):
        y_pred, advs, vf = m_out[0], m_out[1], m_out[2]

        y_true_action = y_true[0].long()
        vf_true = y_true[1]

        # print(f"Dimension ==========+> {y_true.size()}")
        # print(f"y_true={y_true.size()}")
        # print(f"y_pred={y_pred.size()}")
        # print(f"advs={advs.size()}")

        # Execute categorical crossentropy
        neglogp = cross_entrop(
            y_pred,  # Logits from model
            y_true_action,  # True actions chosen
            # sample_weight=advs
        )
        policy_loss = (advs * neglogp).mean()

        cat = Categorical(y_pred)
        entropy_loss = cat.entropy().mean()

        loss_vf = mse(vf, vf_true)

        return policy_loss - entropy_coeff * entropy_loss + coeff_vf * loss_vf

    # Return a function
    return pi_loss
