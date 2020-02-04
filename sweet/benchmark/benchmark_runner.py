"""
This module aims to execute a benchmark between
agents, ML platform, configurations on several environments
"""
import itertools as it
from pathlib import Path

from sweet.common.logging import Logger
from sweet.common.utils import list_to_dict
from sweet.run import make_agent_train_func, make_ml_platform


def benchmark_runner():
    # Benchmark configuration
    agents = ['dqn', 'a2c']
    envs = ['Acrobot-v1', 'CartPole-v1', 'MountainCar-v0']
    ml_platforms = ['tf', 'torch']

    # Create all configuration
    benchmark_combinations = list(it.product(
        envs,
        ml_platforms,
        agents
        )
    )

    for idx, el in enumerate(benchmark_combinations):
        benchmark_combinations[idx] = {
            'env': el[0], 'ml': el[1], 'agent': el[2]}


    # Initialize logger
    run_target_dir = Path('./target/benchmark/')

    logger = Logger(
        "benchmark-runner",
        target_dir=run_target_dir,
        logs_dir='logs',
        tb_dir='tb_events')
    logger.save(
        run_target_dir / 'benchmark_combinations.json',
        list_to_dict(benchmark_combinations)
    )

    for conf in benchmark_combinations:
        logger.info(f"===> Running {conf} <===")
        agent_train_func = make_agent_train_func(conf['agent'])
        ml_platform = make_ml_platform(conf['ml'])

        env_name = conf['env']
        specific_run_target = f"run_{env_name}_{conf['ml']}_{conf['agent']}"

        if conf['agent'] == 'dqn':
            agent_train_func(
                ml_platform=ml_platform,
                env_name=env_name,
                model='dense',
                total_timesteps=5e5,
                lr=0.002,
                targets={
                    'output_dir': run_target_dir / specific_run_target,
                    'models_dir': 'models_checkpoints',
                    'logs_dir': 'logs',
                    'tb_dir': 'tb_events'
                }
            )
        else:  # TODO improve this if/else
            agent_train_func(
                ml_platform=ml_platform,
                env_name=env_name,
                model='dense',
                total_timesteps=5e5,
                lr_actor=0.002,
                lr_critic=0.001,
                targets={
                    'output_dir': run_target_dir / specific_run_target,
                    'models_dir': 'models_checkpoints',
                    'logs_dir': 'logs',
                    'tb_dir': 'tb_events'
                }
            )


if __name__ == '__main__':
    benchmark_runner()
