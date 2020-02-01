
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
```
python3.x -m venv ~/.virtualenvs/sweet/ # or: virtualenv ~/.virtualenvs/sweet/ -p python3
source ~/.virtualenvs/sweet/bin/activate
```
And then, install project dependancies:  
```
make install # or pip install -e .
```

### First execution  

Run a DQN training:  
```
python sweet/agents/dqn/dqn_runner.py # Work in progress
```

## Features, algorithms implemented

| Algorithm     | Implementation status |               |
| ------------- | -------------         | ------------- |
| DQN | <g-emoji class="g-emoji" alias="heavy_check_mark" fallback-src="https://github.githubassets.com/images/icons/emoji/unicode/2714.png">✔️</g-emoji>  |               |
| A2C           | <g-emoji class="g-emoji" alias="heavy_check_mark" fallback-src="https://github.githubassets.com/images/icons/emoji/unicode/2714.png">✔️</g-emoji>  |               |
| PPO           | Soon                  |               |



## Troubleshootings

* Tensorflow 2.x doesn't work with Python 3.8 so far, so only Python versions 3.6 and 3.7 are supported


## History/Author(s)

I started this open-source RL framework in january 2020 as a consecration of my passion for Reinforcement Learning.
Besides coding open-source project, i work for both Airbus and IRT Saint-Exupéry on Earth Observation satellites. Our team is focus on mission planning for satellites and Reinforcement Learning is an approach to solve it. Feel free to discuss with me: [Adrien HADJ-SALAH @linkedin](https://www.linkedin.com/in/adrien-hadj-salah-1b119462/)

## RL related topics

* **What is Reinforcement Learning**
It is supposed that you have knowledge in RL, if it is not the case, take a look to the [spinningup from OpenAI](https://spinningup.openai.com/en/latest/)