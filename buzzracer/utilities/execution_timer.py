# for quick and dirty code profiling
import logging

from time import time, thread_time
from collections import defaultdict

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ExecutionTimer:
    """ Wall-clock runtime profiler"""

    def __init__(self, enable=True):
        self.enabled = enable

        self.start_ts = None
        self.end_ts = None
        self.total_runtime = 0.0
        self.total_count = 0
        self.total_duration = 0.0
        self.child_sections: dict[str, ExecutionTimer] = {}

        """ dict: Tracked variable name -> mean value """
        self.tracked: dict[str, float] = defaultdict(float)
        """ Tracked variable count """
        self.tracked_count: dict[str, int] = defaultdict(int)
        """ Currently active subsession name """
        self.current_subsession = None

    def time(self):
        return time()
        # return thread_time()

    def global_start(self):
        if not self.enabled:
            return
        self.start_ts = self.time()
        return

    def start(self, name=None):
        if not self.enabled:
            return
        if name is None:
            return self.global_start()
        if self.current_subsession is None:
            self.current_subsession = name
            if not name in self.child_sections:
                self.child_sections[name] = ExecutionTimer(enable=True)
            self.child_sections[name].s()

        else:
            return self.child_sections[self.current_subsession].s(name)

    def end(self, name=None):
        if not self.enabled:
            return
        if name is None:
            if self.current_subsession is not None:
                logger.error(f' end() is called before end({self.current_subsession}),'
                             'missed call? check all logic paths')
                self.end(self.current_subsession)

            return self.global_end()

        if self.current_subsession is None:
            logger.error('e() or end() called before s(), timing is corrupt')

        if self.current_subsession == name:
            self.child_sections[name].e()
            self.current_subsession = None
        else:
            try:
                self.child_sections[self.current_subsession].e(name)
            except KeyError:
                logger.error(
                    f'end({name}) is called but no matching start({name}) is called')

    def global_end(self):
        if not self.enabled:
            return

        self.total_duration += self.time() - self.start_ts
        self.total_count += 1
        self.start_ts = None

    def track(self, name, var):
        if not self.enabled:
            return

        self.tracked[name] = self.tracked[name]*self.tracked_count[name]+var
        self.tracked_count[name] += 1
        self.tracked[name] = self.tracked[name] / self.tracked_count[name]

    def s(self, n=None):
        return self.start(n)

    def e(self, n=None):
        return self.end(n)

    def summary(self, prefix='', multiplier=1.0):
        """ Print a formatted summary of execution time, with prefix leading
        Args:
            prefix: text-prefix for controlling subsection indentation
            multiplier: multiplier for percentage spent in subsections
        """
        if not self.enabled:
            return
        if len(self.child_sections) == 0 and prefix == '':
            logger.info('No timed block defined')
            return

        # A long enough field width
        fw = 30

        if len(self.tracked) > 0:
            text = 'Variables'
            logger.info(f'{text:-^{fw}}')
            for key, value in self.tracked.items():
                logger.info(f'{key:<{fw}}{value:>5.2f}')

        if prefix == '':
            text = 'Time'
            logger.info(f'{text:-^{fw}}')
        total_accounted_time = 0.0
        for key, value in self.child_sections.items():
            total_accounted_time += value.total_duration
            frac = value.total_duration / self.total_duration
            logger.info(
                f'{prefix+key:<{fw}}{prefix}{multiplier*frac*100:3.2f}%')
            value.summary(prefix=prefix+'| ', multiplier=multiplier*frac)
        if len(self.child_sections) > 0:
            frac = 1-total_accounted_time/self.total_duration
            logger.info(
                f'{prefix+"Unaccounted":<{fw}}{prefix}{multiplier*frac*100:3.2f}%')

        if prefix == '':
            logger.info(
                f'Avg freq = {self.total_count/self.total_duration:.3f}Hz')
        return


# sample usage
if __name__ == '__main__':
    from time import sleep
    logging.basicConfig(level=logging.INFO)
    # Create an instance of exe_timer for all procedures you want to monitor
    # Initialize with True to enable all functions
    # When you're done analyzing, simply change the argument to False or
    # initialize without an argument, that will cause all methods to return instantly
    t = ExecutionTimer(True)

    # A typical scenario is to find average execution
    # time during several iterations, average exe time will be
    # updated during each iteration
    for i in range(3):
        # Start global timer in the very beginning of the procedure
        t.s()

        # To track an operation, enclose it with INSTANCE.s('identifier')
        # and INSTANCE.e('identifier')
        # s and e are shorthand for start and end
        # each start() must be matched with an end() with identical identifier
        t.s('sleep 2')
        sleep(0.2)
        t.e('sleep 2')

        # We can use the same identifier multiple time to accumulate time under that tag
        for j in range(3):
            t.s('sleep 1*3')
            sleep(0.1)
            t.e('sleep 1*3')

        t.s('sleep with child')
        t.s('child 1')
        sleep(0.01)
        t.e('child 1')

        t.s('child 2')
        sleep(0.02)
        t.s('grandchild')
        sleep(0.01)
        t.e('grandchild')
        t.e('child 2')
        sleep(0.06)
        t.e('sleep with child')

        # not all operations in your procedure will be timed, those not timed are called
        # unaccounted time
        sleep(0.4)

        # it is also possible to track average value of a variable, this is how you do it.
        t.track('var', 5+i/10)

        # at the end of the operation, end the global timer with a matching e()
        t.e()
    # this function prints a summary of everything.
    t.summary()
