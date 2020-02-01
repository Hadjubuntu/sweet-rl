from sweet.interface.ml_platform import MLPlatform


class TorchPlatform(MLPlatform):
    def __init__(self):
        super().__init__('torch')
