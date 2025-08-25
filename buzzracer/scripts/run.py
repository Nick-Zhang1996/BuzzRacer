"""Universal entry point for running simulation or experiments."""
import sys
import os.path
import os
import logging
from threading import Event
from time import time
from xml.dom import minidom

from common import PrintObject, LogObject, ExperimentType

from util.timeUtil import ExecutionTimer
from track import TrackFactory

from car.Car import Car

logger = logging.getLogger('ProfileSteinmerge')
logger.setLevel(logging.INFO)

os.environ['PATH'] = (
    os.environ['PATH'] + ':/usr/local/cuda/bin/')  # enables cuda


class Main(PrintObject, LogObject):
    """Entry point for running simulation or experiments."""

    def __init__(self, config: str):
        LogObject.__init__(self)
        self.basedir = os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))
        self.config_filename = config
        self.experiment_name = config

        self.simulator = None

        # Load config
        # TODO: make this configurable Object
        self.print_ok(' loading settings')
        config = minidom.parse(self.config_filename)
        config_settings = config.getElementsByTagName('settings')[0]
        self.print_ok(' setting main attributes')
        for key, value_text in config_settings.attributes.items():
            setattr(self, key, eval(value_text))
            self.print_info(' main.', key, '=', value_text)

        def get_experiment_type_from_config_settings(config_settings):
            config_experiment_text = config_settings.getElementsByTagName(
                'experiment_type')[0].firstChild.nodeValue
            type_map = {'Simulation': ExperimentType.Simulation,
                        'RealWorld': ExperimentType.Realworld}
            try:
                return type_map[config_experiment_text]
            except KeyError as e:
                raise NameError(
                    f'Unknown experiment type {config_experiment_text},'
                    f'must be one of {type_map.keys}') from e

        self.experiment_type = get_experiment_type_from_config_settings(
            config_settings)

        # Prepare track
        def get_track_from_config(config):
            config_track = config.getElementsByTagName('track')[0]
            track = TrackFactory.build(main=self, config=config_track)
            track.init()
            return track
        self.track = get_track_from_config(config)

        # Prepare cars
        Car.reset()
        config_cars = config.getElementsByTagName('cars')[0]
        self.cars = [Car.Factory(self, config_car)
                     for config_car in config_cars.getElementsByTagName('car')]
        self.print_info(f' total cars: {len(self.cars)}')

        self.timer = ExecutionTimer(True)
        ''' Timer for profiling code '''
        self.new_state_update = Event()
        ''' Event is set when a new state from simulator or Vicon is ready'''
        self.exit_request = Event()
        ''' Flag to quit all child threads gracefully '''
        self.slowdown = Event()
        ''' if set, continue to follow trajectory but set throttle to -0.1
        so we don't leave car uncontrolled at max speed
        currently this is ignored and pressing 'q' the first time will cut motor
        second 'q' will exit program
        '''
        self.slowdown_ts = 0
        ''' Timestamp for when slowdown Event is set '''

        # Load Extensions defined in configs
        self.print_ok('setting up extensions...')
        self.extensions = []
        config_extensions = config.getElementsByTagName('extensions')[0]
        for config_extension in config_extensions.getElementsByTagName(
                'extension'):
            extension_class_name = config_extension.firstChild.nodeValue
            try:
                # pylint: disable-next=exec-used
                exec('from extension import ' + extension_class_name)
            except ImportError:
                self.print_error(f'Cannot import {extension_class_name}')
                raise

            ext = eval(extension_class_name)(self)
            handle_name = ''
            for key, raw in config_extension.attributes.items():
                # handle is the attribute name of this extension
                if key == 'handle':
                    handle_name = raw
                    setattr(self, handle_name, ext)
                    self.print_info('main.' + handle_name + ' = ' +
                                    ext.__class__.__name__)
                else:
                    try:
                        value = eval(raw)
                    except (NameError, SyntaxError):
                        value = raw
                    # all other attributes in config will be added to extension
                    setattr(ext, key, value)
                    self.print_info('main.' + handle_name + '.' + key + ' = ' +
                                    str(value))
        # Some modules depend on other modules to initialize
        # Use pre_init, init, and post_init for crude separation
        for item in self.extensions:
            item.pre_init()
        for car in self.cars:
            car.pre_init()

        for item in self.extensions:
            item.init()
        for car in self.cars:
            car.init()

        for item in self.extensions:
            item.post_init()
        for car in self.cars:
            car.post_init()

    def run(self):
        """Run experiment until user press q in visualization window."""
        self.print_info('running ... press q to quit')
        while not self.exit_request.is_set():
            self.update()

        self.print_info('Exiting ...')
        for car in self.cars:
            car.controller.final()
        for item in self.extensions:
            item.preFinal()
        for item in self.extensions:
            item.final()
        for item in self.extensions:
            item.postFinal()

    @property
    def time(self):
        """Current time, either time() or simulated time if in simulation."""
        if self.experiment_type == ExperimentType.Simulation:
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
        for item in self.extensions:
            t.s(item.name)
            item.pre_update()
            t.e(item.name)

        self.new_state_update.wait()
        self.new_state_update.clear()

        t.s('control')
        for car in self.cars:
            # call controller, send command to car in real experiment
            car.control()
        t.e('control')

        # -- Extension update --
        t.s('update')
        for item in self.extensions:
            item.update()
        t.e('update')
        t.s('post')
        for item in self.extensions:
            item.postUpdate()
        t.e('post')
        t.e()


if __name__ == '__main__':

    # Run default.xml config if none is provided
    name = sys.argv[1] if len(sys.argv) == 2 else 'default'
    config_filename = './configs/' + name + '.xml'

    if os.path.exists(config_filename):
        logger.info('using config %s', config_filename)
    else:
        logger.error(config_filename + '  does not exist!')

    experiment = Main(config_filename)
    experiment.run()
    experiment.timer.summary()
    # experiment.cars[0].controller.p.summary()

    logger.info('program complete')
