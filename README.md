
[![Build status](https://travis-ci.com/Hadjubuntu/sweet-rl.svg?branch=master)](https://travis-ci.com/Hadjubuntu/sweet-rl)<br />
![Sweet-RL](https://raw.githubusercontent.com/Hadjubuntu/sweet-rl/develop/misc/logo.png)

## Why Sweet-RL
 
It exists dozens of Reinforcement Learning frameworks and algorithms implementations.
Yet, most of them suffer of poor modularity and ease of understanding. This is why, I started to implement my own: Sweet-RL.
It's so sweet that you can switch from Tensorflow 2.1 to PyTorch 1.4 with a single configuration line:  
![Sweet-RL](https://raw.githubusercontent.com/Hadjubuntu/sweet-rl/agnostic-ml-platform/misc/ml-platform.png)


## Getting started

### Install sweet-rl  

First, create a virtualenv:  
```bash
python3.x -m venv ~/.virtualenvs/sweet/ 
# or: virtualenv ~/.virtualenvs/sweet/ -p python3
source ~/.virtualenvs/sweet/bin/activate
```
And then, install project dependancies:  
```bash
make install # or pip install -e .
```

### First execution  

Run a DQN training:  
```bash
python -m sweet.run --env=CartPole-V0 --algo=dqn --ml=tf

# Parameters:
#  -h, --help     show this help message and exit
#  --env ENV      Environment to play with (eg. 'CartPole-v0')
#  --algo ALGO    RL agent ('dqn', 'a2c')
#  --ml ML        ML platform ('tf' or 'torch')
#  --model MODEL  Model ('dense', 'conv')
#  --output OUTPUT  Output directory (eg. './target/')

```

### Custom neural network

If you want to specify your own model instead of default ones, take a look to
`sweet.agents.dqn.experiments.train_custom_model`

## Features, algorithms implemented

### Algorithms
| Algorithm     | Implementation status |  ML platform  |
| ------------- | -------------         | ------------- |
| DQN | <g-emoji class="g-emoji" alias="heavy_check_mark" fallback-src="https://github.githubassets.com/images/icons/emoji/unicode/2714.png">✔️</g-emoji>  |  TF2, Torch |
| A2C           | <g-emoji class="g-emoji" alias="heavy_check_mark" fallback-src="https://github.githubassets.com/images/icons/emoji/unicode/2714.png">✔️</g-emoji>  |  TF2, Torch   |
| PPO           | Soon                  |               |


### IO: Logs, model, tensorboard events
Outputs are configurable in training function:
```python
targets: dict = {
        'output_dir': Path('./target/'), # Main directory to store your outputs
        'models_dir': 'models_checkpoints', # Saving models (depending on model_checkpoint_freq)
        'logs_dir': 'logs', # Saving logs (info, debug, errors)
        'tb_dir': 'tb_events' # Saving tensorboard events
}
```

## Benchmark

To reproduce benchmark, execute:
```bash
python -m sweet.benchmark.benchmark_runner
```

## Troubleshootings

* Tensorflow 2.x doesn't work with Python 3.8 so far, so only Python versions 3.6 and 3.7 are supported
* GPU is not used. See https://www.tensorflow.org/install/gpu

## History/Author

I started this open-source RL framework in january 2020, at first to take benefit of tensorflow 2.x readability without sacrifying the performance.
Besides coding open-source project, i work for both Airbus and IRT Saint-Exupéry on Earth Observation satellites. Our team is focus on mission planning for satellites and Reinforcement Learning is an approach to solve it. Feel free to discuss with me: [Adrien HADJ-SALAH @linkedin](https://www.linkedin.com/in/adrien-hadj-salah-1b119462/)

**You are welcome to participate to this project**

## RL related topics

* **What is Reinforcement Learning**
It is supposed that you have knowledge in RL, if it is not the case, take a look to the [spinningup from OpenAI](https://spinningup.openai.com/en/latest/)