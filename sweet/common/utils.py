import tensorflow as tf
import torch


def tf_check_cpu_gpu():
    print(('Is your GPU available for use?\n{0}').format(
        'Yes, your GPU is available: True' if tf.test.is_gpu_available() else 'No, your GPU is NOT available: False'
    ))

    print(('\nYour devices that are available:\n{0}').format(
        [device.name for device in tf.config.experimental.list_physical_devices()]
    ))


def torch_check_cpu_gpu():
    print(f"Using GPU ? {torch.cuda.current_device()}")
    print(f"Cuda available ? {torch.cuda.is_available()}")

