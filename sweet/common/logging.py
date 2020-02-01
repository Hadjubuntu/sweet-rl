import logging
import os
from pathlib import Path
from logging.handlers import RotatingFileHandler


def init_logger(log_in_file=True, target_dir=Path('./target/')):
    """
    Initialize logging activity
    """
    # Force target directory creation if doesn't exist
    os.makedirs(target_dir, exist_ok=True)

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    if log_in_file:
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

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)

    formatter_stream = logging.Formatter(
        '%(name)-12s: %(levelname)-8s %(message)s'
    )
    stream_handler.setFormatter(formatter_stream)
    logger.addHandler(stream_handler)
