import numpy as np
from .base_controller import BaseController

class MyController(BaseController):
    def __init__(self, params=None):
        super().__init__(params)
        self.gain = self.params.get('gain', 0.5)

    def reset(self):
        pass

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        error = target_lataccel - current_lataccel
        steer = self.gain * error
        return float(np.clip(steer, -2.0, 2.0))
    
Controller = MyController

