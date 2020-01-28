
[![Build status](https://travis-ci.com/Hadjubuntu/sweet-rl.svg?branch=master)](https://travis-ci.com/Hadjubuntu/sweet-rl)<br />
![Sweet-RL](https://raw.githubusercontent.com/Hadjubuntu/sweet-rl/develop/misc/logo.png)

## Why Sweet-RL
It exists dozens of Among Reinforcement Learning framework and algorithms implementations. 
Yet, they suffer of poor modularity and ease of understanding. This is why, I started to implement my own: Sweet-RL.
Trying to keep code quality, algorithms performance and diversity as high as possible.

## Getting started

**Install sweet-rl**
Just create a virtualenv and install project:  
```
virtualenv ~/.virtualenvs/sweet/ -p python3
or
python3.x -m venv ~/.virtualenvs/sweet/

then
source ~/.virtualenvs/sweet/bin/activate
```

And install the project:  
```
make install
or
pip install -e .
```

**First execution**
Run a DQN training:  
```
python sweet/agents/dqn/dqn_runner.py # Work in progress
```

## Algorithms implemented

| Algorithm     | Implementation status |               |
| ------------- | -------------         | ------------- |
| DQN           | Ok                    |               |
| A2C           | Ok                    |               |
| PPO           | Soon                  |               |

## Troubleshootings

* Tensorflow 2.x doesn't work with Python 3.8 so far, so only Python versions 3.6 and 3.7 are supported

## History/Author(s)

I started this open-source RL framework in january 2020 as a consecration of my passion for Reinforcement Learning.
Besides coding open-source project, i work for both Airbus and IRT Saint-Exupéry on Earth Observation satellites. Our team is focus on mission planning for satellites and Reinforcement Learning is an approach to solve it. Feel free to discuss with me: [LinkedIn](https://www.linkedin.com/in/adrien-hadj-salah-1b119462/)