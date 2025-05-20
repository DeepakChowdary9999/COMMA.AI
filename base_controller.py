from abc import ABC, abstractmethod

class BaseController(ABC):
    def __init__(self, params=None):
        self.params = params or {}

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        pass