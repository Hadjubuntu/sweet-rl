import tensorflow as tf


class AgentLog():
    """
    Agent logger
    """

    def __init__(self, path):
        """
        """
        self.path = path

    def test(self):
        import numpy as np
        writer = tf.summary.create_file_writer(self.path)

        with writer.as_default():

            for n_iter in range(100):
                tf.summary.scalar('Loss/train', np.random.random(), n_iter)
                tf.summary.scalar('Loss/test', np.random.random(), n_iter)
                tf.summary.scalar('Accuracy/train', np.random.random(), n_iter)
                tf.summary.scalar('Accuracy/test', np.random.random(), n_iter)

    def append(self, metric, value, step):
        """
        """
        writer = tf.summary.create_file_writer(self.path)
        with writer.as_default():
            tf.summary.scalar(metric, value, step=step)
            writer.flush()
