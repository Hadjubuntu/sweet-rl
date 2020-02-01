
import numpy as np
import tensorflow as tf
import tensorflow.keras.losses as kls


def loss_actor_critic(_entropy_coeff=0.0001):
    """
    Loss for actor-part of actor-critic algorithm: policy loss + entropy
    """
    #  FIXME forced to use sparse categorical crossentropy due to y_true
    # coming from actions without one-hot encoding
    weighted_sparse_ce = kls.SparseCategoricalCrossentropy(
            from_logits=True)
    entropy_coeff = _entropy_coeff

    def pi_loss(y_true, y_pred_and_advs):
        y_pred, advs = y_pred_and_advs[0], y_pred_and_advs[1]
        # First, one-hot encoding of true value y_true
        y_true = tf.expand_dims(tf.cast(y_true, tf.int32), axis=1)
        # No-need ? y_true = tf.one_hot(, self.action_size)

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
        print(f"policy_loss = {policy_loss}")
        return policy_loss - entropy_coeff * entropy_loss

    # Return a function
    return pi_loss
