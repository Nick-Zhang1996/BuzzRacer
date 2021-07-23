from math import atan2,radians,degrees,sin,cos,pi,tan,copysign,asin,acos,isnan,exp,pi
class Car:
    # states
    def __init__(self,main):
        Car.main = main
        self.controller = None
        self.throttle = 0.0
        self.steering = 0.0
        self.states = (0,0,0,0,0,0)
        self.debug_dict = {}

    @classmethod
    def Factory(cls, main, config_name, controller, init_states):
        car = cls(main)
        car.controller = controller(car)
        # (x,y,theta,vforward,vsideway=0,omega)
        x,y,heading = init_states
        car.states = (x,y,heading,0,0,0)

        porsche_setting = {'wheelbase':90e-3,
                         'max_steer_angle_left':radians(27.1),
                         'max_steer_pwm_left':1150,
                         'max_steer_angle_right':radians(27.1),
                         'max_steer_pwm_right':1850,
                         'serial_port' : '/dev/ttyUSB0',
                         'optitrack_streaming_id' : 2,
                         'max_throttle' : 0.7}

        lambo_setting = {'wheelbase':98e-3,
                         'max_steer_angle_left':asin(2*98e-3/0.52),
                         'max_steer_pwm_left':1100,
                         'max_steer_angle_right':asin(2*98e-3/0.47),
                         'max_steer_pwm_right':1850,
                         'serial_port' : '/dev/ttyUSB1',
                         'optitrack_streaming_id' : 15,
                         'max_throttle' : 0.5}


        if config_name == "porsche":
            car_setting = porsche_setting
        elif config_name == "porsche_slow":
            car_setting = porsche_setting
            car_setting['max_throttle'] = 0.7
        elif config_name == "lambo":
            car_setting = lambo_setting
        else:
            print_error("Unrecognized car config")
        # max steering is in radians, for vehicle with ackerman steering (inner wheel steer more than outer)
        # steering angle shoud be calculated by arcsin(wheelbase/turning radius), easily derived from non-slipping bicycle model
        # default values are for the MR03 chassis with Porsche 911 GT3 RS body
        car.wheelbase = car_setting['wheelbase']
        car.max_throttle = car_setting['max_throttle']
        car.max_steering_left = car_setting['max_steer_angle_left']
        car.min_pwm_left = car_setting['max_steer_pwm_left']
        car.max_steering_right = car_setting['max_steer_angle_right']
        car.max_pwm_right = car_setting['max_steer_pwm_right']
        return car
