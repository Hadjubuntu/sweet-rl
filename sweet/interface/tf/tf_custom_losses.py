import tensorflow as tf
import tensorflow.keras.losses as kls


def loss_actor_critic(_action_size,  _coeff_vf=0.5, _entropy_coeff=0.01):
    """
    Loss for actor-part of actor-critic algorithm: policy loss + entropy
    """
    cat_crosentropy = kls.CategoricalCrossentropy(
            from_logits=True)
    entropy_coeff = _entropy_coeff
    action_size = _action_size
    coeff_vf = _coeff_vf

    def pi_loss(y_true, m_out):
        y_pred, advs, vf = m_out[0], m_out[1], m_out[2]

        y_true_action = y_true[0]
        vf_true = tf.cast(y_true[1], tf.float32)

        # First, one-hot encoding of true value y_true
        y_true_action = tf.expand_dims(
            tf.cast(y_true_action, tf.int32),
            axis=1
        )
        y_true_action = tf.one_hot(y_true_action, depth=action_size)

        # Execute categorical crossentropy
        neglogp = cat_crosentropy(
            y_true_action,  # True actions chosen
            y_pred,  # Logits from model
            # sample_weight=advs
        )
        policy_loss = tf.reduce_mean(advs * neglogp)

        entropy_loss = kls.categorical_crossentropy(
            y_pred, y_pred,
            from_logits=True
        )

        loss_vf = kls.mean_squared_error(vf, vf_true)

        return policy_loss - entropy_coeff * entropy_loss + coeff_vf * loss_vf

    # Return a function
    return pi_loss
