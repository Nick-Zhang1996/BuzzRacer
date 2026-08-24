"""Universal entry point for running simulation or experiments."""
import os.path
import os
from time import time, perf_counter
from xml.dom import minidom
import multiprocessing as mp
import ctypes

from buzzracer.common import PrintObject, LogObject, ExperimentType, Config, get_logger
from buzzracer.types import Control, CartesianState, StateTiming, ControlTiming
from buzzracer.utilities.execution_timer import ExecutionTimer
from buzzracer.tracks.track_factory import TrackFactory
from buzzracer.cars.car import Car
from buzzracer.extensions.extension import Extension
from buzzracer.controllers.controller import Controller

logger = get_logger(__name__)


class MainConfig:
    """ Config class for Main"""

    def __init__(self, config_filename):
        self.dt = 0.01
        self.multiprocess = True
        self.config_filename = config_filename

        dom = minidom.parse(config_filename)
        self.dom = dom
        dom_settings = dom.getElementsByTagName('settings')[0]
        logger.info('Setting main attributes')
        for key, value_text in dom_settings.attributes.items():
            if not hasattr(self, key):
                raise AttributeError(f'Unknown attribute {key}={value_text}')
            # pylint: disable-next=eval-used
            setattr(self, key, eval(value_text))
            logger.info(f' {__name__}.{key}.{value_text}')

        def get_experiment_type_from_config_settings(dom_settings):
            exp_type_text = dom_settings.getElementsByTagName(
                'experiment_type')[0].firstChild.nodeValue
            type_map = {
                'Simulation': ExperimentType.Simulation,
                'RealWorld': ExperimentType.Realworld
            }
            try:
                return type_map[exp_type_text]
            except KeyError as e:
                raise NameError(
                    f'Unknown experiment type {exp_type_text},'
                    f'must be one of {list(type_map.keys())}') from e

        dom_cars: Config = dom.getElementsByTagName('cars')[0]
        self.car_configs = [
            val for val in dom_cars.getElementsByTagName('car')
        ]

        self.dom_track = dom.getElementsByTagName('track')[0]
        self.dom_extensions = dom.getElementsByTagName('extensions')[0]
        self.experiment_type = get_experiment_type_from_config_settings(
            dom_settings)
        self.experiment_name = os.path.basename(config_filename).split('.')[0]
        self.car_count = len(self.car_configs)


class MainState:
    """ Process safe state """

    def __init__(self, main_config: MainConfig):
        car_count = main_config.car_count
        self.experiment_type = main_config.experiment_type
        self.new_state_update = mp.Event()
        ''' Event is set when a new state from simulator or Vicon is ready'''
        self.exit_request = mp.Event()
        ''' Flag to quit gracefully '''
        self.slowdown = mp.Event()
        ''' if set, continue to follow trajectory but set throttle to -0.1
        so we don't leave car uncontrolled at max speed
        currently this is ignored and pressing 'q' the first time will cut motor
        second 'q' will exit program
        '''
        self.breakpoint = mp.Event()
        """ Set by visualization listing on key stroke 'b'. Can be cleared for debugging """
        self.use_optitrack_observations = mp.Event()
        """ Set when controllers should consume raw OptiTrack observations. """
        self.car_observations = mp.Array(CartesianState, car_count, lock=False)
        self.car_states = mp.Array(CartesianState, car_count, lock=False)
        self.car_states_event = [mp.Event() for _ in range(car_count)]
        self.car_states_first_available = mp.Event()
        """ Car specific event for new state available, set by main"""
        self.car_state_timing = mp.Array(StateTiming, car_count, lock=False)
        self.car_control = mp.Array(Control, car_count, lock=False)
        self.car_control_event = [mp.Event() for _ in range(car_count)]
        """ Car specific event for new control available, set by controller"""
        self.car_control_timing = mp.Array(ControlTiming, car_count, lock=False)
        self.car_target_v = mp.Array('d', car_count, lock=False)
        self._time = mp.Value(ctypes.c_double, 0.0)
        """ Shared timestamp for the latest published state snapshot. """

    def publish_time(self, sim_t=None):
        """Publish the time associated with the latest shared state snapshot."""
        if self.experiment_type == ExperimentType.Simulation:
            if sim_t is None:
                raise ValueError('sim_t must be provided in simulation mode')
            now = sim_t
        else:
            now = time()
        with self._time.get_lock():
            self._time.value = now

    @property
    def time(self):
        """Timestamp associated with the latest shared state snapshot."""
        with self._time.get_lock():
            return self._time.value


class Main(PrintObject, LogObject):
    """Entry point for running simulation or experiments."""

    def __init__(self, config_filename: str):
        LogObject.__init__(self)
        self.config = MainConfig(config_filename)
        self.simulator = None

        # Prepare track

        def get_track_from_dom(dom_track):
            track = TrackFactory.build(config=dom_track)
            track.init()
            return track

        self.track = get_track_from_dom(self.config.dom_track)

        # Prepare cars
        Car.reset(self)
        self.cars: list[Car] = [
            Car.Factory(cfg) for cfg in self.config.car_configs
        ]
        self.print_info(f' total cars: {len(self.cars)}')
        self.state = MainState(self.config)
        self.state.publish_time(0.0 if self.config.experiment_type == ExperimentType.Simulation
                                else None)

        self.timer = ExecutionTimer(True, clock=perf_counter, clock_name='wall')
        ''' Timer for profiling code '''
        self.latency_timer = ExecutionTimer(True, clock=perf_counter, clock_name='wall')
        """Timer for inter-process state-to-command latency in milliseconds."""
        self.new_state_update = self.state.new_state_update
        ''' Event is set when a new state from simulator or Vicon is ready'''
        self.exit_request = self.state.exit_request
        ''' Flag to quit all child threads gracefully '''
        self.slowdown = self.state.slowdown
        ''' if set, continue to follow trajectory but set throttle to -0.1
        so we don't leave car uncontrolled at max speed
        currently this is ignored and pressing 'q' the first time will cut motor
        second 'q' will exit program
        '''

        # Load Extensions defined in configs
        Extension.load(self, self.config, self.config.dom_extensions)

        # Some modules depend on other modules to initialize
        # Use pre_init, init, and post_init for crude separation
        Extension.pre_init_all()
        for car in self.cars:
            car.pre_init()
        Extension.init_all()
        for car in self.cars:
            car.init()

        Extension.post_init_all()
        for car in self.cars:
            car.post_init()

        if not hasattr(self, 'planner'):
            self.planner = None

        if self.config.multiprocess:
            self.child_processes = []
            for car in self.cars:
                p = mp.Process(target=Controller.process_fun,
                               args=(self.state, car.id, car.param, self.track,
                                     car.controller.__class__,
                                     car.controller.config,
                                     car.controller.state,
                                     None if self.planner is None else self.planner.state)
                               )
                p.start()
                self.child_processes.append(p)
            logger.info("Multiprocess enabled")

    def run(self):
        """Run experiment until user press q in visualization window."""
        self.print_info('running ... press q to quit')
        while not self.exit_request.is_set():
            self.update()

        self.print_info('Exiting ...')

        if self.config.multiprocess:
            for p in self.child_processes:
                p.join()

        for car in self.cars:
            car.controller.final()
        Extension.pre_final_all()
        Extension.final_all()
        Extension.post_final_all()
        if len(self.latency_timer.tracked) > 0:
            logger.info('State-to-command latency summary [ms]')
            self.latency_timer.summary()

    @property
    def time(self):
        """Current time, either time() or simulated time if in simulation."""
        return self.state.time

    def update(self, ):
        """Run the control/visualization update.

        This should be called in a loop(while not self.exit_request.is_set())
        continuously, without delay.

        In simulation, this is called with evenly spaced time.

        In real experiment, this is called after a new vicon update is available.
        When a new vicon/optitrack state is available, vi.newState() is set and
        client (this function) need to unset that event

        """
        t = self.timer
        # -- Extension update --
        t.s()
        t.s('pre update')
        Extension.pre_update_all(t)
        t.e('pre update')

        t.s('wait new state update')  # 60% Time
        self.new_state_update.wait()
        self.new_state_update.clear()
        if self.config.multiprocess:
            if not getattr(self, 'shared_state_published_immediately', False):
                for i, car in enumerate(self.cars):
                    self.state.car_states[i] = car.state
                    self.state.car_state_timing[i] = StateTiming(
                        car._latency_state_seq,
                        car._latency_udp_rx_ts,
                        car._latency_rigid_body_ts,
                        car._latency_state_set_ts)
                self.state.publish_time(
                    self.simulator.sim_t if self.config.experiment_type == ExperimentType.Simulation
                    else None)
                for i, _ in enumerate(self.cars):
                    self.state.car_states_event[i].set()
                    # logger.info(f'{i=}, {self.state.car_states[i]=}')
        else:
            self.state.publish_time(
                self.simulator.sim_t if self.config.experiment_type == ExperimentType.Simulation
                else None)
        if not (self.config.multiprocess and self.state.use_optitrack_observations.is_set()):
            self.state.car_states_first_available.set()
        t.e('wait new state update')

        t.s('control')
        # Call controllers
        for i, car in enumerate(self.cars):
            car = self.cars[i]
            # t.s(car.param.name)
            if self.config.multiprocess:
                hardware_consumes_control = (
                    car.consumes_multiprocess_control
                    and self.config.experiment_type == ExperimentType.Realworld
                )
                if not hardware_consumes_control:
                    self.state.car_control_event[i].wait(0.1)
                    self.state.car_control_event[i].clear()
                    Controller.apply_multiprocess_control(car, self.state, i)
            else:
                # Call controller one by one
                result = car.controller.control(
                    car.state, car.param, self.track, car.controller.config,
                    car.controller.state, self.state, i)
                if len(result) == 4:
                    control, _, controller_state, _ = result
                else:
                    control, _, controller_state = result
                car.controller.state = controller_state
                car.steering = control.steering

                car.throttle = 0.0 if self.state.slowdown.is_set(
                ) else control.throttle
            # t.e(car.param.name)
        t.e('control')

        # -- Extension update --
        Extension.update_all(t)
        Extension.post_update_all(t)
        t.e()
