''' Record speed profile in debug_dict '''
import matplotlib.pyplot as plt

from buzzracer.extensions.extension import Extension, ExtensionConfig, ExtensionState


class SpeedTrackerConfig(ExtensionConfig):
    def __init__(self, main_config):
        super().__init__(main_config)
        self.car_id = 0


@Extension.register('speed_tracker', SpeedTrackerConfig, ExtensionState)
class SpeedTracker(Extension):
    ''' Record speed profile in debug_dict '''

    def __init__(self, config, state):
        super().__init__(config, state)
        self.target_v_vec = []
        self.actual_v_vec = []
        self.throttle_vec = []

    def update(self):
        config = self.config
        self.target_v_vec.append(Extension.main.state.car_target_v[config.car_id])
        self.actual_v_vec.append(Extension.main.state.car_states[config.car_id].v_forward)
        self.throttle_vec.append(Extension.main.state.car_control[config.car_id].throttle)

    def final(self):
        plt.plot(self.target_v_vec, label='target')
        plt.plot(self.actual_v_vec, label='actual')
        plt.plot(self.throttle_vec, label='throttle')
        plt.legend()
        plt.show()
