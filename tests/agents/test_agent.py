from sweet.agents.agent import Agent

import numpy as np


class MockAgent(Agent):
    def __init__(self):
        print("mock")

    def act(self, obs):
        pass


def test_discount_rewards():
    """
    Check that agent compute good discounted reward from returns,
    dones status and gamma factor
    """
    agent = MockAgent()

    n = 10

    x = np.ones(n)
    dones = np.zeros(n)
    expected_returns = np.arange(start=1, stop=11)[::-1]
    computed_returns = agent.discount_with_dones(x, dones, gamma=1.0)

    assert (computed_returns == expected_returns).all()
