''' Base class for all car controllers '''
from __future__ import annotations
from typing import TYPE_CHECKING
import logging
from time import process_time

from buzzracer.common import LogObject, set_config_attr, LoggingFilter
from buzzracer.utilities.execution_timer import ExecutionTimer
from buzzracer.types import CartesianState, Control
if TYPE_CHECKING:
    from buzzracer.main import MainState, MainConfig
    from buzzracer.cars.car_param import CarParam
    from buzzracer.tracks.track import Track

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addFilter(LoggingFilter(interval=1.0))


class ControllerConfig:
    """ Base class for Car Controller Config.
    Declares and stores all non-mutable configuration parameters.
    Controller.factory will create an instance of this class, 
    set all attribtues in config xml to the instance, then set it as controller.config
    If an attribute is missing from declaration in __init__, it will not be configurable
    in config xml. This avoids misspelt parameters.
    Car Controller subclasses should subclass and add new configs. 
    """

    # pylint:disable-next=unused-argument
    def __init__(self, main_config: MainConfig, car_param: CarParam):
        """ Use planner instead of main.track """
        self.planner = False


class ControllerState:
    """ Base class for Car Controller State.
    Declares and stores all mutable state variables for a controller.
    Controller.factory will create an instance of this class, then set it to controller.state
    Car Controller subclasses should subclass and add new states. 
    The main purpose of this class is to separate the implementation and mutable data.
    To support multi-process operation, the controller state will be moved between processes. 
    Encapsulating the mutable states avoids unnecessary copying.
    Controller intended for multiprocess operation should define a static function control()
    that depends on State instance, but not self.
    """

    def __init__(self, config: ControllerConfig):
        pass


class Controller(LogObject):
    registry = {}
    config_registry = {}
    state_registry = {}

    @staticmethod
    def register(config_cls, state_cls):
        """Decorator to add a controller class to the registry."""

        def wrapper(cls):
            name = cls.__name__
            Controller.registry[name] = cls
            Controller.config_registry[name] = config_cls
            Controller.state_registry[name] = state_cls
            print(f'registered {name}')
            return cls

        return wrapper

    @staticmethod
    def factory(name, main_config: MainConfig, car_param: CarParam,
                config_minidom):
        """ Create Controller instance, corresponding Config, and State instance.
        Args:
            main: Class name of the controller
            car_param: car parameters,
            config_minidom: Minidom object for controller config.
        Returns:
            (controller, config, state)"""
        controller_cls = Controller.registry[name]
        config_cls = Controller.config_registry[name]
        state_cls = Controller.state_registry[name]
        config = config_cls(main_config, car_param)
        config = set_config_attr(config_minidom, config)
        state = state_cls(config)
        controller = controller_cls()
        controller.config = config
        controller.state = state
        return (controller, config, state)

    def __init__(self):
        self.config = None
        self.state = None
        LogObject.__init__(self)

    def pre_init(self):
        return

    def post_init(self):
        return

    def init(self):
        return

    def final(self):
        """called at end of program, override to show statistics."""
        return

    @staticmethod
    def control(car_state: CartesianState,
                car_params,
                track,
                controller_config,
                controller_state,
                main_state,
                car_index,
                planner_state=None):
        ''' Given state of the vehicle and an instance of track,
        provide throttle and steering output
        Args:
          state: CartesianState (x,y,heading,v_forward,v_sideway,omega)
          car_params: CarParams, parameters of the car
          track: track object, can be RCPTrack or skidpad

        Outputs:
          control: Control(steering, throttle)
          valid:    If the car can be controlled here, false if too far off reference.
                    If this is false, then throttle will also be set to 0
          state: updated controller state
        '''
        del car_state, car_params, track, controller_config, main_state, car_index, planner_state
        ctrl = Control(steering=0, throttle=0)
        valid = False
        return (ctrl, valid, controller_state)

    @staticmethod
    def process_fun(main_state: MainState, car_index: int,
                    car_params: CarParam, track: Track, controller_cls,
                    controller_config, controller_state, planner_state=None):
        """ Entry point for multi process control. """
        timer = ExecutionTimer(enable=True, clock=process_time, clock_name='process_cpu')
        try:
            while not main_state.exit_request.is_set():
                timer.s()
                v_override = 0.0 if main_state.slowdown.is_set() else None
                controller_state.v_override = v_override

                timer.s('wait for state')
                new_state = main_state.car_states_event[car_index].wait(0.1)
                timer.e('wait for state')
                if not new_state:
                    timer.e()
                    continue
                main_state.car_states_event[car_index].clear()

                timer.s('control')
                control, valid, state, msg = controller_cls.control(
                    main_state.car_states[car_index], car_params, track,
                    controller_config, controller_state, main_state, car_index, planner_state)
                timer.e('control')

                if not valid:
                    logger.warning('Invalid control for %s, %s', car_params.name, msg)
                controller_state = state

                timer.s('publish control')
                main_state.car_control[car_index] = control
                main_state.car_control_event[car_index].set()
                timer.e('publish control')
                timer.e()
        finally:
            logger.info('Controller CPU-time summary for %s', car_params.name)
            timer.summary()
