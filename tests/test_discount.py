def discount_with_dones(rewards, dones, gamma):
    """
    Compute discounted rewards
    """
    discounted = []
    r = 0
    for reward, done in zip(rewards[::-1], dones[::-1]):
        r = reward + gamma*r*(1.-done) # fixed off by one bug
        discounted.append(r)
    return discounted[::-1]

import numpy as np

x=np.ones(10)
dones=np.zeros(10)
print(discount_with_dones(x, dones, gamma=1.0))