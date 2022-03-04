from common import *
import serial
from math import atan2,radians,degrees,sin,cos,pi,tan,copysign,asin,acos,isnan,exp,pi
class Car:
    car_count = 0
    cars = []
    # states
    def __init__(self,main):
        Car.main = main
        self.controller = None
        self.throttle = 0.0
        self.steering = 0.0
        #x,y,heading,v_forward,v_sideways(left positive),omega(angular speed,turning to left positive)
        self.states = (0,0,0,0,0,0)
        self.debug_dict = {}
        self.car_interface = None

    def init(self):
        if (Car.main.experiment_type == ExperimentType.Realworld):
            self.initSerialInterface()
        self.controller.init()

    def initSerialInterface(self):
        try:
            self.car_interface = serial.Serial(self.serial_port,115200, timeout=0.001,writeTimeout=0)
        except (FileNotFoundError,serial.serialutil.SerialException):
            print_error("[Car]: interface %s not found"%self.serial_port)
            exit(1)

    def control(self):
        if (self.controller is None):
            self.throttle = 0.0
            self.steering = 0.0
        else:
            # TODO: address when controller can't find a valid solution
            self.controller.control()
            #print_info("[Car]: "+"T=%4.1f, S=%4.1f"%(self.throttle, degrees(self.steering)))
            #print_info(self.states)

        if (Car.main.slowdown.is_set()):
            self.throttle = 0.0
        if (Car.main.experiment_type == ExperimentType.Realworld):
            self.actuate()

    def actuate(self):
        if not (self.car_interface is None):
            self.car_interface.write((str(self.mapdata(self.steering, self.max_steering_left,-self.max_steering_right,self.min_pwm_left,self.max_pwm_right))+","+str(self.mapdata(self.throttle,-1.0,1.0,1900,1100))+'\n').encode('ascii'))
            return True
        else:
            return False

    # provide direct pwm
    def actuatePWM(self,steeringPWM,throttlePWM):
        if not (self.car_interface is None):
            self.car_interface.write((str(int(steeringPWM))+","+str(int(throttlePWM))+'\n').encode('ascii'))
            return True
        else:
            return False
    def __del__(self):
        if ((not self.serial_port is None) and (not self.car_interface is None) and self.car_interface.is_open):
            self.car_interface.close()
    def mapdata(self,x,a,b,c,d):
        y=(x-a)/(b-a)*(d-c)+c
        return int(y)

    @classmethod
    def reset(cls):
        cls.cars = []
        cls.car_count = 0
    @classmethod
    def Factory(cls, main, config_name, controller, init_states):
        car = cls(main)
        # (x,y,theta,vforward,vsideway=0,omega)
        x,y,heading,v_forward = init_states
        car.states = (x,y,heading,v_forward,0,0)

        porsche_setting = {'wheelbase':90e-3,
                         'max_steer_angle_left':radians(27.1),
                         'max_steer_pwm_left':1150,
                         'max_steer_angle_right':radians(27.1),
                         'max_steer_pwm_right':1850,
                         'serial_port' : '/dev/ttyUSB0',
                         'optitrack_streaming_id' : 2,
                         #'optitrack_streaming_id' : 998,
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
        car.serial_port = car_setting['serial_port']
        car.optitrack_id = car_setting['optitrack_streaming_id']
        car.id = Car.car_count
        if not controller is None:
            car.controller = controller(car)

        # physics properties
        # Defaults for when a specific car instance is not speciied
        car.L = 0.09
        car.lf = 0.04824
        car.lr = car.L - car.lf

        car.Iz = 417757e-9
        car.m = 0.1667

        Car.cars.append(car)
        Car.car_count += 1
        return car
