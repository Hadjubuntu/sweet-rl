import tensorflow as tf

class AgentLog():
    def __init__(self, path):
        self.path = path

    def append(self, metric, value, step):
        writer = tf.summary.create_file_writer(self.path)
        with writer.as_default():
            tf.summary.scalar(metric, value, step=step)
            writer.flush()


    


