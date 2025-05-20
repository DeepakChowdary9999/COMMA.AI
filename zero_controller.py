from controllers.base_controller import BaseController

class Controller(BaseController):
    def reset(self):
        pass

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        return 0.0