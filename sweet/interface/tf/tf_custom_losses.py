
import numpy as np
import tensorflow as tf
import tensorflow.keras.losses as kls


def loss_actor_critic(_action_size, _entropy_coeff=0.0001):
    """
    Loss for actor-part of actor-critic algorithm: policy loss + entropy
    """
    weighted_sparse_ce = kls.CategoricalCrossentropy(
            from_logits=True)
    entropy_coeff = _entropy_coeff
    action_size = _action_size

    def pi_loss(y_true, y_pred_and_advs):
        y_pred, advs = y_pred_and_advs[0], y_pred_and_advs[1]
        # First, one-hot encoding of true value y_true
        y_true = tf.expand_dims(tf.cast(y_true, tf.int32), axis=1)
        y_true = tf.one_hot(y_true, depth=action_size)

        # Execute categorical crossentropy
        neglogp = weighted_sparse_ce(
            y_true,  # True actions chosen
            y_pred,  # Logits from model
            # sample_weight=advs
        )
        policy_loss = tf.reduce_mean(advs * neglogp)

        entropy_loss = kls.categorical_crossentropy(
            y_pred, y_pred,
            from_logits=True
        )

        return policy_loss - entropy_coeff * entropy_loss

    # Return a function
    return pi_loss
