# Sweet-RL
![alt text](https://raw.githubusercontent.com/Hadjubuntu/sweet-rl/master/misc/logo.png)
A nice and sweet reinforcement learning framework

## Install sweet-rl

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

Run a training:  
```
python sweet/agents/dqn/dqn_runner.py # Work in progress
```


## Troubleshootings

* Tensorflow 2.x doesn't work with Python 3.8 so far, so only Python versions 3.6 and 3.7 are supported