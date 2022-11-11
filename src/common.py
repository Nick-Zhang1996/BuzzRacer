import numpy as np
from enum import Enum, auto
import inspect


class ExperimentType(Enum):
    Simulation = auto()
    Realworld = auto()
    Replay = auto()

class PrintObject:
    debug = False
    def __init__(self):
        #print_ok(self.prefix() + "in use")
        #self.debug = False
        pass

    def print_debug_enable(self):
        self.debug = True
    def print_debug_disable(self):
        self.debug = False

    def prefix(self):
        return "["+self.__class__.__name__+"]: "

    def print_error(self, *message):
        print('\033[91m', self.prefix(),'ERROR ', *message, '\033[0m')
        raise RuntimeError

    def print_ok(self, *message):
        print('\033[92m',self.prefix(), *message, '\033[0m')

    def print_debug(self, *message):
        # yellow
        if (self.debug):
            print('\033[93m',self.prefix(), inspect.stack()[1][3],*message, '\033[0m')

    def print_warning(self, *message):
        # yellow
        #print('\033[93m',self.prefix(), *message, '\033[0m')
        # red
        print('\033[91m',self.prefix(), 'WARNING: ', *message, '\033[0m')

    def print_info(self, *message):
        print('\033[96m',self.prefix(), *message, '\033[0m')



# if variables are declared in a subclass for readability,
# then the contructor of this method needs to be called AFTER those definitions
# so they don't get overridden
class ConfigObject(PrintObject):
    def __init__(self,config):
        self.config = config
        if config is None:
            self.print_warning('no config available')
            return
        self.print_ok("setting " + config.firstChild.nodeValue + " attributes")
        # load config parameters
        for key,value_text in config.attributes.items():
            try:
                value = eval(value_text)
            except NameError:
                value = value_text
            setattr(self,key,value)
            self.print_info(config.firstChild.nodeValue, ".",key,'=',value_text)

# ----------

def print_error(*message):
    print('\033[91m', 'ERROR ', *message, '\033[0m')
    raise RuntimeError
def print_ok(*message):
    print('\033[92m', *message, '\033[0m')

def print_warning(*message):
    # yellow
    #print('\033[93m', *message, '\033[0m')
    # red
    print('\033[91m', 'WARNING: ', *message, '\033[0m')

def print_info(*message):
    print('\033[96m', *message, '\033[0m')

def ndarray(x):
    return np.asarray(x, dtype=np.float64)

def angular_difference(a, b):
    diff = a - b
    angle_diff = diff - np.floor(diff / (2 * np.pi)) * 2 * np.pi
    while angle_diff > np.pi:
        angle_diff -= 2 * np.pi
    while angle_diff < -np.pi:
        angle_diff += 2 * np.pi
    return angle_diff

