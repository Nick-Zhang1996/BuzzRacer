class CarController:
    def __init__(self, car):
        self.car = car
        self.track = car.main.track

    # return control signals
    # throttle, steering
    def control(self):
        throttle = 0.0
        steering = 0.0
        return (throttle, steering)
