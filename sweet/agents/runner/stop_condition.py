from abc import ABC, abstractmethod

class StopCond(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def iterate(self, **kwargs):
        pass

    @abstractmethod
    def reset(self):
        pass

class NstepsStopCond(StopCond):
    def __init__(self, nsteps=128):
        super().__init__()
        self.nsteps = nsteps
        self.current_nstep = 0

    def reset(self):
        self.current_nstep = 0

    def iterate(self, **kwargs):
        self.current_nstep += 1        
        return self.current_nstep < self.nsteps

class EpisodeDoneStopCond(StopCond):
    def __init__(self):
        super().__init__()
    
    def reset(self):
        pass
    
    def iterate(self, **kwargs):
        return not kwargs['done']