from sweet.common.schedule import ConstantSchedule, LinearSchedule
"""
Test capacity to schedule a variable value along training
"""


def test_const_schedule():
    schedule = ConstantSchedule(const=0.5)

    for ti in range(0, 100):
        assert schedule.value(t=ti) == 0.5


def test_linear_schedule():
    # Test trivial linear schedule
    schedule = LinearSchedule(
        schedule_timesteps=100,
        initial_value=1.0,
        final_value=0.0)

    for ti in range(0, 100):
        assert schedule.value(t=ti) == (100.0 - ti) / 100.0

    # Test linear schedule with non-zero ending value
    schedule = LinearSchedule(
        schedule_timesteps=100,
        initial_value=1.0,
        final_value=0.5)
    assert schedule.value(t=100) == 0.5
