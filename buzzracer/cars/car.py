''' Defines the interface for working with physical and simulated cars'''
from math import radians, degrees, asin

from buzzracer.common import PrintObject, LogObject, ExperimentType, get_logger
from buzzracer.controllers.car_controller import CarController
from buzzracer.types import CartesianState


logger = get_logger('Car')


class Car(PrintObject, LogObject):
    ''' Base class for various types of cars,
    Subclasses should implement communication details to interact with different
    types of physical car.'''
    car_count = 0
    ''' Total number of cars'''
    cars = []
    ''' List of cars '''

    def __init__(self, main):
        LogObject.__init__(self)
        self.main = main
        self.controller: CarController = None
        self._throttle = 0.0
        self._steering = 0.0
        self.state = CartesianState(0,0,0,0,0,0)

        # default values, will be overridden in config
        self.max_throttle = 1.0
        self.min_throttle = -1.0
        self.max_steering_left = radians(26.1)
        self.max_steering_right = radians(26.1)
        self.debug_dict = {}

    @property
    def throttle(self):
        return self._throttle

    @throttle.setter
    def throttle(self, val):
        val = val if val < self.max_throttle else self.max_throttle
        val = val if val > self.min_throttle else self.min_throttle
        self._throttle = val

    @property
    def steering(self):
        return self._steering

    @steering.setter
    def steering(self, val):
        val = val if val < self.max_steering_left else self.max_steering_left
        val = val if val > -self.max_steering_right else -self.max_steering_right
        self._steering = val

    def pre_init(self):
        self.controller.pre_init()

    def post_init(self):
        self.controller.post_init()

    def init(self):
        ''' Initialization for cars.
        This will be run after initialization for all other extensions have concluded
        '''
        if self.main.experiment_type is ExperimentType.Realworld:
            self.init_hardware()
        self.controller.init()

    def init_hardware(self):
        ''' Initialize code that require hardware initializations here '''

    def actuate(self):
        ''' Send control commands to the car'''

    def control(self):
        if self.controller is None:
            self.throttle = 0.0
            self.steering = 0.0
        else:
            self.controller.control()
            self.text_logger.debug('T=%4.1f, S=%4.1f deg'%(self.throttle, degrees(self.steering)))

        if self.main.slowdown.is_set():
            self.throttle = 0.0
        if self.main.experiment_type == ExperimentType.Realworld:
            self.actuate()

    @classmethod
    def reset(cls):
        cls.cars = []
        cls.car_count = 0

    @classmethod
    def Factory(cls, main, config):
        try:
            hardware_class_text = config.getElementsByTagName(
                'hardware')[0].firstChild.nodeValue
            # pylint: disable-next=exec-used
            exec('from buzzracer.cars import '+hardware_class_text)
        except IndexError:
            logger.warning('no hardware specified')

        config_controller = config.getElementsByTagName('controller')[0]
        controller_class_text = config_controller.getElementsByTagName('type')[
            0].firstChild.nodeValue

        try:
            init_states_text = config.getElementsByTagName('init_states')[0].firstChild.nodeValue
            init_states = eval(init_states_text)
        except IndexError:
            logger.warning('Car: no initial state specified, using track default')
            init_states = (*main.track.start_pos, main.track.start_dir, 0.1)

        config_name = config.getElementsByTagName('config_name')[0].firstChild.nodeValue
        # pylint: disable-next=exec-used
        exec(f'from buzzracer.controllers import {controller_class_text}')
        controller = eval(controller_class_text)

        car = eval(hardware_class_text)(main)

        # (x,y,theta,vforward,vsideway=0,omega)
        x, y, heading, v_forward = init_states
        car.state = (x, y, heading, v_forward, 0, 0)

        porsche = {'wheelbase': 90e-3,
                   'max_steering_left': radians(27.1),
                   'max_steer_pwm_left': 1150,
                   'max_steering_right': radians(27.1),
                   'max_steer_pwm_right': 1850,
                   'serial_port': '/dev/ttyUSB0',
                   'optitrack_streaming_id': 2,
                   # 'optitrack_streaming_id' : 998,
                   'max_throttle': 1.0,
                   'min_throttle': -1.0,
                   'rendering': 'data/porsche_orange.png'}

        porsche_slow = {'wheelbase': 90e-3,
                        'max_steering_left': radians(27.1),
                        'max_steer_pwm_left': 1150,
                        'max_steering_right': radians(27.1),
                        'max_steer_pwm_right': 1850,
                        'serial_port': '/dev/ttyUSB0',
                        'optitrack_streaming_id': 2,
                        'max_throttle': 1.0,
                        'min_throttle': -1.0,
                        'rendering': 'data/porsche_orange.png'}

        lambo = {'wheelbase': 98e-3,
                 'max_steering_left': asin(2*98e-3/0.52),
                 'max_steer_pwm_left': 1100,
                 'max_steering_right': asin(2*98e-3/0.47),
                 'max_steer_pwm_right': 1850,
                 'serial_port': '/dev/ttyUSB1',
                 'optitrack_streaming_id': 15,
                 'max_throttle': 1.0,
                 'min_throttle': -1.0,
                 'rendering': 'data/porsche_green.png'}

        orca = {'wheelbase': 0.029+0.033,
                'width': 0.03,
                'rendering': 'data/porsche_green.png'}

        # TODO render audi
        audi_11 = {'optitrack_streaming_id': 998,
                   'ip': '192.168.10.11',
                         'max_steering_left': radians(26.1),
                         'max_steering_right': radians(26.1),
                         'max_throttle': 1.0,
                         'min_throttle': -1.0,
                         'rendering': 'data/porsche_green.png'}

        audi_12 = {'optitrack_streaming_id': 1005,
                   'ip': '192.168.10.12',
                         'max_steering_left': radians(26.1),
                         'max_steering_right': radians(26.1),
                         'max_throttle': 1.0,
                         'min_throttle': -1.0,
                         'rendering': 'data/porsche_orange.png'}

        sim_green = {'optitrack_streaming_id': 1005,
                     'ip': '192.168.10.12',
                     'max_steering_left': 13.0,
                     'max_steering_right': 13.0,
                     'max_throttle': 10.0,
                     'min_throttle': -10.0,
                     'max_v': 4.0,
                     'max_ax': 10.0,
                     'max_ay': 13.0,
                     'rendering': 'data/porsche_green.png'}

        sim_red = {'optitrack_streaming_id': 1005,
                   'ip': '192.168.10.12',
                         'max_steering_left': 9.0,
                         'max_steering_right': 9.0,
                         'max_throttle': 13.0,
                         'min_throttle': -13.0,
                         'max_v': 4.0,
                         'max_ax': 13.0,
                         'max_ay': 9.0,
                         'rendering': 'data/porsche_orange.png'}

        car.params = eval(config_name)

        if not controller is None:
            car.controller = controller(car, config_controller)

        car.init_param()

        car.id = Car.car_count
        Car.cars.append(car)
        Car.car_count += 1

        return car
