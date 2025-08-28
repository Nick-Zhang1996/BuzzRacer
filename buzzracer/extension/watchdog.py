''' Extension to terminate experiment if stuck or something goes wrong'''

from buzzracer.extension.extension import Extension
from buzzracer.car.car import Car


class Watchdog(Extension):
    ''' Extension to terminate experiment if stuck or something goes wrong'''
    def __init__(self):
        Extension.__init__(self, handle_name='watchdog')
        self.triggered = False
        ''' A halting condition has been detected by Watchdog'''
        self.trigger_reason: str = ''
        ''' Short description of why Watchdog was triggered'''
        self.car_is_on_track: dict[Car, bool] = {car:True for car in self.main.cars}

    def post_update(self):
        msg = []
        for car in self.main.cars:
            x = car.states[0]
            y = car.states[1]
            vf = car.states[3]

            # if car is outside track, halt
            if (self.main.track.is_outside((x, y))):
                self.car_is_on_track[car] = False
                self.triggered = True
                msg.append('car outside track, terminating experiment')

            # if car is too slow, halt
            if (vf < 0.05):
                self.triggered = True
                msg.append('car stopped, terminating experiment')

            # if no new laps in a long time, halt
            if (self.main.simulator.sim_t - car.laptimer.last_lap_ts > 20):
                self.triggered = True
                msg.append('No new laps detected for %.2f s, terminating experiment' % (car.laptimer.last_laptime))

            # if laptime is unreasonable, halt
            if (car.laptimer.new_lap.is_set() and car.lap_count > 0):
                if (car.laptimer.last_laptime < 2.0):
                    self.triggered = True
                    msg.append('unreasonable laptime: %.2f, terminating experiment' % (car.laptimer.last_laptime))

        combined_msg = ' , and '.join(msg)
        self.print_warning(combined_msg)
        self.trigger_reason = combined_msg
        if (self.triggered):
            self.main.exit_request.set()