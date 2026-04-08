''' Defines the interface for working with physical and simulated cars'''
from __future__ import annotations
import logging

# pylint: disable-next=unused-import
from math import degrees, radians

from buzzracer.common import PrintObject, LogObject, ExperimentType
from buzzracer.types import CartesianState, CurvilinearState
from buzzracer.cars.car_param import CarParam, CarConfig

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)


class Car(PrintObject, LogObject):
    ''' Base class for various types of cars,
    Subclasses should implement communication details to interact with different
    types of physical car.'''
    car_count = 0
    ''' Total number of cars'''
    cars = []
    """ All cars, this include cars of different subclass.
    If a subclass needs a list of cars of that specific subclass, it must overwrite this attribute"""
    main = None
    """ Access to main"""
    registry = {}
    """ Registry of all Car subclasses"""

    def __init__(self):
        LogObject.__init__(self)
        self.controller = None
        self._throttle = 0.0
        self._steering = 0.0
        self.state = CartesianState(0, 0, 0, 0, 0, 0)
        self.param: CarParam

        # default values, will be overridden in config
        self.max_throttle = 1.0
        self.min_throttle = -1.0
        self.debug_dict = {}

    @staticmethod
    def register(cls):
        """Decorator to add a car class to the registry."""
        name = cls.__name__
        Car.registry[name] = cls
        _logger.debug('Registered %s' % {name})
        return cls

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
        val = val if val < self.param.max_steer_left else self.param.max_steer_left
        val = val if val > -self.param.max_steer_right else -self.param.max_steer_right
        self._steering = val

    def pre_init(self):
        self.controller.pre_init()

    def post_init(self):
        self.controller.post_init()

    def init(self):
        ''' Initialization for cars.
        This will be run after initialization for all other extensions have concluded
        '''
        if Car.main.config.experiment_type is ExperimentType.Realworld:
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
            _logger.debug('T=%4.1f, S=%4.1f deg' %
                          (self.throttle, degrees(self.steering)))

        if Car.main.state.slowdown.is_set():
            self.throttle = 0.0
        if Car.main.config.experiment_type == ExperimentType.Realworld:
            self.actuate()

    @classmethod
    def reset(cls, main):
        Car.main = main
        Car.cars = []
        Car.car_count = 0

    @classmethod
    def Factory(cls, config_minidom):
        cm = config_minidom
        from buzzracer.controllers.controller import Controller
        try:
            car_cls_text = cm.getElementsByTagName('hardware')[0].firstChild.nodeValue
            car_cls = Car.registry[car_cls_text]
        except IndexError:
            _logger.warning('No hardware specified')

        config_ctrl = cm.getElementsByTagName('controller')[0]
        controller_class_text = config_ctrl.getElementsByTagName('type')[0].firstChild.nodeValue

        # pylint: disable-next=unused-variable
        def curv(s, n, phi, v):
            curv = CurvilinearState(s, n, phi, v)
            cart = Car.main.track.curv_to_cart(curv)
            return cart.to_tuple()[:4]
        try:
            init_states_text = cm.getElementsByTagName('init_states')[0].firstChild.nodeValue
            init_states = eval(init_states_text)
        except IndexError:
            _logger.warning('Car: no initial state specified, using track default')
            init_states = (*Car.main.track.data.start_pos, Car.main.track.data.start_dir, 0.1)

        config_name = cm.getElementsByTagName('config_name')[0].firstChild.nodeValue

        car = car_cls()
        # (x,y,theta,vforward,vsideway=0,omega)
        x, y, heading, v_forward = init_states
        car.state = CartesianState(x=x,
                                   y=y,
                                   heading=heading,
                                   v_forward=v_forward,
                                   v_sideway=0,
                                   omega=0)

        car.param = getattr(CarConfig, config_name).value

        controller, controller_config, controller_state = Controller.factory(
            controller_class_text, Car.main.config, car.param,
            config_ctrl)

        if not controller is None:
            car.controller = controller
        else:
            _logger.warning('No controller for %s', config_name)

        car.id = Car.car_count
        Car.cars.append(car)
        Car.car_count += 1

        return car
