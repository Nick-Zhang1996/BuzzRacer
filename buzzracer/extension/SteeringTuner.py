from extension.Extension import Extension
import matplotlib.pyplot as plt
class SteeringTuner(Extension):
    def __init__(self,main):
        Extension.__init__(self,main)
    def final(self):
        for car in self.main.cars:
            plt.plot(car.steering_requested_vec,'--')
            plt.plot(car.steering_measured_vec)
            plt.show()
