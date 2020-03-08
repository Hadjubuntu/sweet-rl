from sweet.agents.a2c.train import learn
from sweet.interface.tf.tf_platform import TFPlatform
from sweet.interface.torch.torch_platform import TorchPlatform

# Just test distribution for continuous action space env
if __name__ == "__main__":
    learn(
        ml_platform=TFPlatform,
        env_name='MountainCarContinuous-v0'
    )
