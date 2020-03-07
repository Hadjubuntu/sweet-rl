import tensorflow as tf
import tensorflow.keras.losses as kls


def loss_actor_critic(_dist,  _coeff_vf=0.5, _coeff_entropy=0.001):
    """
    Loss for actor-part of actor-critic algorithm: policy loss + entropy
    """
    dist = _dist
    coeff_vf = _coeff_vf
    coeff_entropy = _coeff_entropy

    def pi_loss(y_true, m_out):
        y_pred, advs, vf = m_out[0], m_out[1], m_out[2]

        y_true_action = y_true[0]
        vf_true = tf.cast(y_true[1], tf.float32)

        # First, one-hot encoding of true value y_true
        y_true_action = tf.expand_dims(
            tf.cast(y_true_action, tf.int32),
            axis=1
        )
        y_true_action = tf.one_hot(y_true_action, depth=dist.n())

        # Compute negative log-likelihood of logits
        neglogp = dist.neglogp(
            y_true_action,  # True actions chosen
            y_pred,  # Logits from model
            # sample_weight=advs
        )
        policy_loss = tf.reduce_mean(advs * neglogp)

        entropy_loss = dist.entropy(y_pred)

        loss_vf = kls.mean_squared_error(vf, vf_true)

        return policy_loss - coeff_entropy * entropy_loss + coeff_vf * loss_vf

    # Return a function
    return pi_loss
