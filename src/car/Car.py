from common import *
import serial
from math import atan2,radians,degrees,sin,cos,pi,tan,copysign,asin,acos,isnan,exp,pi
class Car(PrintObject):
    car_count = 0
    cars = []
    # initialization for variables common to all subclass
    def __init__(self,main):
        self.main = main
        self.controller = None
        self._throttle = 0.0
        self._steering = 0.0
        #x,y,heading,v_forward,v_sideways(left positive),omega(angular speed,turning to left positive)
        self.states = (0,0,0,0,0,0)
        # default values, will be overridden
        self.max_throttle = 1.0
        self.max_steering_left = radians(26.1)
        self.max_steering_right = radians(26.1)
        self.debug_dict = {}

    @property
    def throttle(self):
        return self._throttle
    @throttle.setter
    def throttle(self,val):
        val = val if val < self.max_throttle else self.max_throttle
        val = val if val > -1.0 else -1.0
        self._throttle = val

    @property
    def steering(self):
        return self._steering
    @steering.setter
    def steering(self,val):
        val = val if val < self.max_steering_left else self.max_steering_left
        val = val if val > -self.max_steering_right else -self.max_steering_right
        self._steering = val


    # this will be run when initialization for all other extensions(visualization, track, vision tracking, simulation etc)
    # have concluded
    def init(self):
        if (self.main.experiment_type == ExperimentType.Realworld):
            self.initHardware()
        self.controller.init()

    # initialize code that require hardware here
    def initHardware(self):
        pass

    def actuate(self):
        #self.print_info(self.throttle, self.steering)
        pass

    def control(self):
        if (self.controller is None):
            self.throttle = 0.0
            self.steering = 0.0
        else:
            # TODO: address when controller can't find a valid solution
            self.controller.control()
            #print_info("[Car]: "+"T=%4.1f, S=%4.1f"%(self.throttle, degrees(self.steering)))
            #print_info(self.states)

        if (self.main.slowdown.is_set()):
            self.throttle = 0.0
        if (self.main.experiment_type == ExperimentType.Realworld):
            self.actuate()

    @classmethod
    def reset(cls):
        cls.cars = []
        cls.car_count = 0

    @classmethod
    def Factory(cls, main, config):
        # TODO error handling, it's ok there's no hardware
        try:
            hardware_class_text = config.getElementsByTagName('hardware')[0].firstChild.nodeValue
            exec('from car import '+hardware_class_text)
        except IndexError:
            self.print_warning('no hardware specified')

        config_controller = config.getElementsByTagName('controller')[0]
        controller_class_text = config_controller.getElementsByTagName('type')[0].firstChild.nodeValue

        try:
            init_states_text = config.getElementsByTagName('init_states')[0].firstChild.nodeValue
            init_states = eval(init_states_text)
        except IndexError:
            print_warning('Car: no initial state specified')
            init_states = (0,0,0,0)

        config_name = config.getElementsByTagName('config_name')[0].firstChild.nodeValue
        exec('from controller import '+controller_class_text)
        controller = eval(controller_class_text)
        config_name = config_name

        car = eval(hardware_class_text)(main)

        # (x,y,theta,vforward,vsideway=0,omega)
        x,y,heading,v_forward = init_states
        car.states = (x,y,heading,v_forward,0,0)

        porsche = {'wheelbase':90e-3,
                         'max_steer_angle_left':radians(27.1),
                         'max_steer_pwm_left':1150,
                         'max_steer_angle_right':radians(27.1),
                         'max_steer_pwm_right':1850,
                         'serial_port' : '/dev/ttyUSB0',
                         'optitrack_streaming_id' : 2,
                         'width' : 0.0461,
                         #'optitrack_streaming_id' : 998,
                         'max_throttle' : 1.0,
                         'rendering' : 'data/porsche_orange.png'}


        porsche_slow = {'wheelbase':90e-3,
                         'max_steer_angle_left':radians(27.1),
                         'max_steer_pwm_left':1150,
                         'max_steer_angle_right':radians(27.1),
                         'max_steer_pwm_right':1850,
                         'serial_port' : '/dev/ttyUSB0',
                         'optitrack_streaming_id' : 2,
                         'width' : 0.0461,
                         'max_throttle' : 1.0,
                         'rendering' : 'data/porsche_orange.png'}

        lambo = {'wheelbase':98e-3,
                         'max_steer_angle_left':asin(2*98e-3/0.52),
                         'max_steer_pwm_left':1100,
                         'max_steer_angle_right':asin(2*98e-3/0.47),
                         'max_steer_pwm_right':1850,
                         'serial_port' : '/dev/ttyUSB1',
                         'optitrack_streaming_id' : 15,
                         'width' : 0.0461,
                         'max_throttle' : 1.0,
                         'rendering' : 'data/porsche_green.png'}

        orca = {          'wheelbase':0.029+0.033,
                         'width' : 0.03,
                         'rendering' : 'data/porsche_green.png'}

        # TODO render audi
        audi_11 = {'wheelbase':98e-3,
                         'optitrack_streaming_id' : 998,
                         'ip' : '192.168.0.11',
                         'max_steer_angle_left':radians(26.1),
                         'max_steer_angle_right':radians(26.1),
                         'max_throttle' : 1.0,
                         'width' : 0.0461,
                         'rendering' : 'data/porsche_green.png'}

        audi_12 = {'wheelbase':98e-3,
                         'optitrack_streaming_id' : 1005,
                         'ip' : '192.168.0.12',
                         'max_steer_angle_left':radians(26.1),
                         'max_steer_angle_right':radians(26.1),
                         'max_throttle' : 1.0,
                         'width' : 0.0461,
                         'rendering' : 'data/porsche_green.png'}

        car.params = eval(config_name)
        #print_error("Unrecognized car config")

        if not controller is None:
            car.controller = controller(car,config_controller)

        # ----
        car.initParam()

        car.id = Car.car_count
        Car.cars.append(car)
        Car.car_count += 1

        return car
