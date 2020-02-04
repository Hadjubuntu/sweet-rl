import tensorflow as tf
import logging
import os
import json

from pathlib import Path
from logging.handlers import RotatingFileHandler
from collections import defaultdict


class Logger():
    def __init__(
        self,
        scope="root",
        target_dir=Path('./target/'),
        logs_dir=Path('logs'),
        tb_dir=Path('tb_events'),
        log_in_file=True,
    ):
        """
        Create logger

        Parameters
        ----------
            scope: str
                Logger name
            target_dir: Path
                Directory of logs (needed if log_in_file is true)
            log_in_file: boolean
                True if you want to export logs in text file
        """
        self.target_dir = target_dir

        self.logger = logging.getLogger(scope)
        self.init_console_logger()
        if log_in_file:
            self.init_file_logger(self.target_dir / logs_dir)

        self.name2val = defaultdict()
        self.step = 0

        self.writer = tf.summary.create_file_writer(
            str(self.target_dir/tb_dir)
        )

    def info(self, s):
        self.logger.info(s)

    def debug(self, s):
        self.logger.debug(s)

    def error(self, s):
        self.logger.error(s)

    def record_tabular(self, k, v):
        self.name2val[k] = v

    def save(self, path: Path, data: dict):
        """
        Save a dictionnary into a file.
        Useful to save configuration
        """
        self.info(data)
        os.makedirs(path.parent, exist_ok=True)

        # Force conversion of object (classes) to string
        # preventing json dumps error
        for k in data.keys():
            if not isinstance(data[k], str):
                data[k] = str(data[k])

        with open(path, 'w') as f:
            json.dump(data, f, indent=4)

    def dump_tabular(self):
        # Tensorboard export
        self.writekvs(self.name2val)
        self.name2val.clear()

    def init_file_logger(self, target_dir):
        """
        Initialize file logger
        """
        # Force target directory creation if doesn't exist
        os.makedirs(target_dir, exist_ok=True)
        logger = self.logger
        formatter = logging.Formatter(
                '%(asctime)s :: %(levelname)s :: %(message)s'
        )
        file_handler = RotatingFileHandler(
            target_dir / 'activity.log',
            'a',
            10000000,
            1
        )

        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    def init_console_logger(self):
        """
        Initialize console logger
        """
        logger = self.logger
        logger.setLevel(logging.DEBUG)

        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)

        formatter_stream = logging.Formatter(
            '%(name)-12s: %(levelname)-8s %(message)s'
        )
        stream_handler.setFormatter(formatter_stream)
        logger.addHandler(stream_handler)

    def writekvs(self, kvs, scope='tb_logs'):
        # TODO add key to scope
        with tf.name_scope(scope):
            with self.writer.as_default():
                for k, v in kvs.items():
                    self.logger.info(f"{k}={v}")
                    
                    if isinstance(v, float) or isinstance(v, int):
                        tf.summary.scalar(k, v, self.step)

        self.step += 1
        self.writer.flush()

    def close(self):
        if self.writer:
            self.writer.Close()
            self.writer = None
