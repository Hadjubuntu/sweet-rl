from gym.envs.registration import register
"""
Registering env allows to create env from gym.make function
env = gym.make('name_of_env')
"""
register(
    id='basic-v0',
    entry_point='sweet.envs:BasicEnv',
)