''' Record speed profile in debug_dict '''
import matplotlib.pyplot as plt

from buzzracer.extensions.extension import Extension



class SpeedTracker(Extension):
    ''' Record speed profile in debug_dict '''

    def __init__(self):
        Extension.__init__(self, 'speed_tracker')
        self.target_v_vec = []
        self.actual_v_vec = []
        self.throttle_vec = []
        for car in Extension.main.cars:
            if car.param.name == 'corvette_17':
                self.car_id = car.id
                print('found target car')
                break

    def update(self):
        self.target_v_vec.append(Extension.main.state.car_target_v[self.car_id])
        self.actual_v_vec.append(Extension.main.state.car_states[self.car_id].v_forward)
        self.throttle_vec.append(Extension.main.state.car_control[self.car_id].throttle)
    def final(self):
        plt.plot(self.target_v_vec, label='target')
        plt.plot(self.actual_v_vec, label='actual')
        plt.plot(self.throttle_vec, label='throttle')
        plt.legend()
        plt.show()

