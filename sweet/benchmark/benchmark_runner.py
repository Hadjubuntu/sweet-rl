"""
This module aims to execute a benchmark between
agents, ML platform, configurations on several environments
"""
import itertools as it


def benchmark_runner():
    agents = ['dqn', 'a2c']
    envs = ['CartPole-V0']
    ml_platforms = ['tf', 'torch']

    benchmark_combinations = list(it.product(
        envs,
        ml_platforms,
        agents
        )
    )
    print(benchmark_combinations)


if __name__ == '__main__':
    benchmark_runner()
