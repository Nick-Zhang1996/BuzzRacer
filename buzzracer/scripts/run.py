"""Universal entry point for running simulation or experiments."""
import sys
import os.path
import os
import logging
from time import time
from xml.dom import minidom
import multiprocessing as mp

from buzzracer.common import PrintObject, LogObject, ExperimentType, Config, BASEDIR, get_logger
from buzzracer.types import Control, CartesianState
from buzzracer.utilities.execution_timer import ExecutionTimer
from buzzracer.tracks.track_factory import TrackFactory
from buzzracer.cars.car import Car
from buzzracer.extensions.extension import Extension
from buzzracer.controllers.car_controller import CarController


logger = get_logger('Run')
logging.basicConfig(level=logging.INFO)

os.environ['PATH'] = (
    os.environ['PATH'] + ':/usr/local/cuda/bin/')  # enables cuda


class MainState:
    """ Process safe state """

    def __init__(self, car_count):
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
        self.car_states = mp.Array(CartesianState, car_count, lock=False)
        self.car_states_event = [mp.Event() for _ in range(car_count)]
        """ Car specific event for new state available, set by main"""
        self.car_control = mp.Array(Control, car_count, lock=False)
        self.car_control_event = [mp.Event() for _ in range(car_count)]
        """ Car specific event for new control available, set by controller"""


class MainConfig:
    def __init__(self, config_filename):
        self.dt = 0.01
        self.multiprocess = False

        dom = minidom.parse(config_filename)
        self.dom = dom
        dom_settings = dom.getElementsByTagName('settings')[0]
        logger.info('Setting main attributes')
        for key, value_text in dom_settings.attributes.items():
            if not hasattr(self, key):
                raise AttributeError(f'Unknown attribute {key}={value_text}')
            setattr(self, key, eval(value_text))
            logger.info(f' {__name__}.{key}.{value_text}')

        def get_experiment_type_from_config_settings(dom_settings):
            exp_type_text = dom_settings.getElementsByTagName(
                'experiment_type')[0].firstChild.nodeValue
            type_map = {'Simulation': ExperimentType.Simulation,
                        'RealWorld': ExperimentType.Realworld}
            try:
                return type_map[exp_type_text]
            except KeyError as e:
                raise NameError(
                    f'Unknown experiment type {exp_type_text},'
                    f'must be one of {list(type_map.keys())}') from e

        dom_cars: Config = dom.getElementsByTagName('cars')[0]
        self.car_configs = [
            val for val in dom_cars.getElementsByTagName('car')]

        self.dom_track = dom.getElementsByTagName('track')[0]
        self.experiment_type = get_experiment_type_from_config_settings(
            dom_settings)
        self.experiment_name = os.path.basename(config_filename).split('.')[0]


class Main(PrintObject, LogObject):
    """Entry point for running simulation or experiments."""

    def __init__(self, config_filename: str):
        LogObject.__init__(self)
        self.config = MainConfig(config_filename)
        self.simulator = None

        # Prepare track

        def get_track_from_dom(dom_track):
            track = TrackFactory.build(main=self, config=dom_track)
            track.init()
            return track
        self.track = get_track_from_dom(self.config.dom_track)

        # Prepare cars
        Car.reset()
        self.cars: list[Car] = [Car.Factory(
            self, cfg) for cfg in self.config.car_configs]
        self.print_info(f' total cars: {len(self.cars)}')
        self.state = MainState(len(self.cars))

        self.timer = ExecutionTimer(True)
        ''' Timer for profiling code '''
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
        Extension.load(self, self.config.dom)

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

        if self.config.multiprocess:
            self.child_processes = []
            for car in self.cars:
                p = mp.Process(target=CarController.process_fun,
                               args=(self.state, car.id, car.params, self.track,
                                     car.controller.__class__, car.controller.config, car.controller.state))
                p.start()
                self.child_processes.append(p)

    def run(self):
        """Run experiment until user press q in visualization window."""
        self.print_info('running ... press q to quit')
        while not self.exit_request.is_set():
            self.update()

        self.print_info('Exiting ...')

        if self.config.multiprocess:
            for p in self.child_processes:
                p.join()

        # TODO does this still work for multiprocess?
        for car in self.cars:
            car.controller.final()
        Extension.pre_final_all()
        Extension.final_all()
        Extension.post_final_all()

    @property
    def time(self):
        """Current time, either time() or simulated time if in simulation."""
        if self.config.experiment_type == ExperimentType.Simulation:
            return self.simulator.sim_t
        else:
            return time()

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
            for i, car in enumerate(self.cars):
                self.state.car_states[i] = car.state
                self.state.car_states_event[i].set()
        t.e('wait new state update')

        t.s('control')
        # Call controllers
        for i, car in enumerate(self.cars):
            car = self.cars[i]
            t.s(car.params.name)
            if self.config.multiprocess:
                # Wait for controller process to complete
                self.state.car_control_event[i].wait(0.1)
                self.state.car_control_event[i].clear()
                car.steering = self.state.car_control[i].steering
                car.throttle = 0.0 if self.state.slowdown.is_set() else self.state.car_control[i].throttle
            else:
                # Call controller one by one
                control, _, controller_state = car.controller.control(
                    car.state, car.params, self.track, car.controller.config, car.controller.state, self.state)
                car.controller.state = controller_state
                car.steering = control.steering
                
                car.throttle = 0.0 if self.state.slowdown.is_set() else control.throttle
            t.e(car.params.name)
        t.e('control')

        # -- Extension update --
        Extension.update_all(t)
        Extension.post_update_all(t)
        t.e()


if __name__ == '__main__':

    # Run default.xml config if none is provided
    name = sys.argv[1] if len(sys.argv) == 2 else 'default'
    config_filename = os.path.join(
        BASEDIR, 'buzzracer', 'configs', f'{name}.xml')

    if os.path.exists(config_filename):
        logger.info('using config %s', config_filename)
    else:
        logger.error(config_filename + '  does not exist!')

    experiment = Main(config_filename)
    experiment.run()
    experiment.timer.summary()
    # experiment.cars[0].controller.p.summary()

    logger.info('program complete')
