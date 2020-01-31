from abc import ABC, abstractmethod


class Schedule(ABC):
    @abstractmethod
    def value(self, t):
        """
        Value of the schedule at time t

        Parameters
        ----------
            t: float
                Iteration number
        Returns
        ----------
            float: Value of schedule
        """
        raise NotImplementedError()


class ConstantSchedule(Schedule):
    def __init__(self, const):
        """
        Constant schedule

        Parameters
        ----------
            const: float
                Constant value
        """
        self.const = const

    def value(self, t):
        return self.const


class LinearSchedule(Schedule):
    def __init__(self, schedule_timesteps, initial_value, final_value):
        """
        Linear schedule decrease value from initial value to min value along timesteps
        """
        self.schedule_timesteps = schedule_timesteps
        self.initial_value = initial_value
        self.final_value = final_value

    def value(self, t):
        fraction = min((self.schedule_timesteps - t) /
                       self.schedule_timesteps, 1.0)
        return self.final_value + fraction * \
            (self.initial_value - self.final_value)
