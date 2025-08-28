''' log experiment parameter for batch experiment '''
from xml.dom import minidom
import os

from buzzracer.extension.extension import Extension


class ConfigLogger(Extension):
    ''' Log experiment wide metadata and statistics'''

    def __init__(self):
        Extension.__init__(self, 'config_logger')

    def init(self):
        config = minidom.parse(self.main.config_filename)
        config_extensions = config.getElementsByTagName('extensions')[0]
        for config_extension in config_extensions.getElementsByTagName(
                'extension'):
            if config_extension.getAttribute('handle') == 'simulator':
                # simulator specific logging
                pass

    def post_final(self):
        # stuff to log down
        entry = []
        # experiment name, this is config folder name (e.g. Cu_a_param_sweep)
        # config file name, (e.g. Cu_a_param_sweep/exp48.xml)
        # log name
        # laps
        # enable_cvar
        # cvar_A
        # cvar_a
        # cvar_Cu
        # laptime_mean
        # laptime_stddev
        # boundary violation
        # obstacle violation
        # labels = ('experiment name'
        #           'config file name,'
        #           'log name ,'
        #           'laps , Qop1, Qop2,'
        #           'start_lead_i_j, end_lead_i_j,'
        #           'laptime_mean , laptime_stddev ,'
        #           'boundary violation , obstacle violation')
        entry.append(self.main.experiment_name)
        entry.append(self.main.config_filename)
        entry.append(self.main.logger.logFilename)
        entry.append(self.main.lap_counter.total_laps)

        # retrieve config params
        #config_filename = self.main.config_filename
        #config = minidom.parse(config_filename)
        #config_cars = config.getElementsByTagName('cars')[0]
        #config_car = config_cars.getElementsByTagName('car')[0]
        #config_controller = config_car.getElementsByTagName('controller')[0]
        #attrs = config_controller.attributes.items()

        # start position (delta s)
        # aggressiveness
        # entry.append( config_controller.getAttribute('Qop1') )
        # entry.append( config_controller.getAttribute('Qop2') )
        entry.append(self.main.cars[0].controller.Qop1)
        entry.append(self.main.cars[0].controller.Qop2)
        entry.append(self.main.cars[0].controller.start_lead_i_j)
        entry.append(self.main.cars[0].controller.end_lead_i_j)

        # these may not be available if watchdog is triggered
        if (not self.main.watchdog.triggered):
            entry.append(self.main.car_laptime_mean[0])
            entry.append(self.main.cars[0].total_boundary_collision)
            entry.append(self.main.car_laptime_mean[1])
            entry.append(self.main.cars[1].total_boundary_collision)
            entry.append(self.main.opponent_collision_count)
        else:
            entry.append(-1)
            entry.append(-1)
            entry.append(-1)
            entry.append(-1)
            entry.append(-1)

        log_name = os.path.join(self.main.logger.logFolder, 'textlog.txt')
        with open(log_name, 'a', encoding='utf8') as f:
            # f.write(labels)
            # f.write('\n')
            self.print_info('text log at ' + log_name)
            text_entry = [str(item) for item in entry]
            f.write(','.join(text_entry))
            f.write('\n')
