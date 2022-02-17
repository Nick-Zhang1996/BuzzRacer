from common import PrintObject
class CarController(PrintObject):
    def __init__(self, car):
        self.car = car
        self.track = car.main.track

    def init(self):
        return

    # return control signals
    # throttle, steering
    def control(self):
        throttle = 0.0
        steering = 0.0
        return (throttle, steering)
