from sweet.run import run
import itertools as it
from pathlib import Path

def test_run():
    # Only few steps for test
    timesteps = 128

    # Compute all sub testing conf
    envs = ['CartPole-v0']
    ml_platforms = ['torch', 'tf']
    agents = ['dqn', 'a2c']

    test_combinations = list(it.product(
        envs,
        ml_platforms,
        agents
        )
    )

    # Finally test them all
    for conf in test_combinations:
        env_str, ml_platform_str, agent_str = conf
        run(
            agent_str,
            ml_platform_str,
            env_str,
            'dense',
            timesteps,
            './target/')


