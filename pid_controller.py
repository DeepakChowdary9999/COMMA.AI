from controllers.base_controller import BaseController

class Controller(BaseController):
    def __init__(self, params=None):
        super().__init__(params)
        p = params or {}
        self.kp = p.get('kp', 1.0)
        self.ki = p.get('ki', 0.0)
        self.kd = p.get('kd', 0.0)
        self.integral = 0.0
        self.prev_error = 0.0

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        error = target_lataccel - current_lataccel
        dt = 0.1
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 0 else 0.0
        self.prev_error = error
        return self.kp * error + self.ki * self.integral + self.kd * derivative
