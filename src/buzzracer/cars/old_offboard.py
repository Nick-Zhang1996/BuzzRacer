''' Subclass of Car for radio-controlled Mini-z'''
import serial

from buzzracer.cars.car import Car, CarParam


class OldOffboard(Car):
    car_count = 0
    cars = []
    # states

    def __init__(self, main):
        self.car_interface = None
        Car.__init__(self, main)
        self.param: CarParam | None = None

    def init_hardware(self):
        try:
            self.car_interface = serial.Serial(
                self.param.serial_port, 115200, timeout=0.001, writeTimeout=0)
        except (FileNotFoundError, serial.serialutil.SerialException):
            self.print_error('interface %s not found' % self.param.serial_port)
            exit(1)

    def actuate(self):
        Car.actuate(self)
        steering_pwm = int(self.mapdata(self.steering,
                                        self.max_steering_left,
                                        -self.max_steering_right,
                                        self.param.min_pwm_left,
                                        self.param.max_pwm_right))
        throttle_pwm = self.mapdata(self.throttle, -1.0, 1.0, 1900, 1100)
        if not self.car_interface is None:
            self.car_interface.write(f'{steering_pwm},{throttle_pwm}\n'.encode('ascii'))
            return True
        else:
            return False

    def actuate_p_w_m(self, steeringPWM, throttlePWM):
        if not self.car_interface is None:
            self.car_interface.write(
                (str(int(steeringPWM))+','+str(int(throttlePWM))+'\n').encode('ascii'))
            return True
        else:
            return False

    def __del__(self):
        if ((not self.param.serial_port is None)
            and (not self.car_interface is None)
                and self.car_interface.is_open):
            self.car_interface.close()

    def mapdata(self, x, a, b, c, d):
        y = (x-a)/(b-a)*(d-c)+c
        return int(y)
