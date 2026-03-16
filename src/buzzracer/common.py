''' Project-wide Common functions and classes'''
import os
import logging

from enum import Enum, auto

import numpy as np

BASEDIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
''' BuzzRacer project base folder.'''


class Config:
    ''' Dummy class for type checking'''


class ExperimentType(Enum):
    Simulation = auto()
    Realworld = auto()
    Replay = auto()


class LogObject:
    """ Base class for logging attributes of a class recursively.

    To use this, user need to add LogObject as a parent class for the class
    they wish to log.

    During use, populate a self.debug_dict of type Dictionary then add
    the class's log as key/value pairs. If the class has member
    variables that need to be logged, declare them as LogObject and
    handle their debug_dict by themselves. There is no need for a class
    to handle its member variable's debug_dict, populate_log will take
    care of that

    """

    def __init__(self):
        self.debug_dict = {}

    def pre_update(self):
        self.debug_dict = {}

    @staticmethod
    def populate_log(root, logged=None):
        """build a tree of debug_dict."""
        if logged is None:
            logged = set()
        debug_dict = root.debug_dict
        logged.add(root)
        for att in dir(root):
            item = getattr(root, att)
            if (isinstance(item, LogObject) and not item in logged):
                debug_dict[att] = LogObject.populate_log(item, logged)

        return debug_dict


# ANSI color codes
RESET = "\033[0m"
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[96m"


class ColoredFormatter(logging.Formatter):
    COLORS = {
        logging.DEBUG: BLUE,
        logging.INFO: GREEN,
        logging.WARNING: YELLOW,
        logging.ERROR: RED,
        logging.CRITICAL: RED,
    }

    def format(self, record):
        # Short prefix: just class name or logger name
        prefix = f"[{record.name}]"
        levelname = record.levelname

        # Apply color based on level
        color = self.COLORS.get(record.levelno, RESET)
        message = super().format(record)

        return f"{color}{prefix} {levelname}: {message}{RESET}"


def get_logger(name: str, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)  # set min log level

    # avoid duplicate handlers if already configured
    if not logger.handlers:
        ch = logging.StreamHandler()
        formatter = ColoredFormatter(
            "%(message)s")  # message only, prefix handled manually
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger


class PrintObject:
    text_logger: logging.Logger

    def __init_subclass__(cls):
        """Called when a subclass is created; set up a class-specific logger."""
        super().__init_subclass__()
        cls.text_logger = get_logger(cls.__name__)

    @classmethod
    def print_debug_enable(cls):
        cls.text_logger.setLevel(logging.DEBUG)

    @classmethod
    def print_debug_disable(cls):
        cls.text_logger.setLevel(logging.INFO)

    @classmethod
    def prefix(cls):
        return '[' + cls.__class__.__name__ + ']: '

    @classmethod
    def print_error(cls, *message):
        cls.text_logger.error(*message)

    @classmethod
    def print_info(cls, *message):
        # green
        cls.text_logger.info(*message)

    @classmethod
    def print_ok(cls, *message):
        # green
        cls.text_logger.info(*message)

    @classmethod
    def print_debug(cls, *message):
        # light blue
        cls.text_logger.debug(*message)

    @classmethod
    def print_warning(cls, *message):
        cls.text_logger.warning(*message)


# if variables are declared in a subclass for readability,
# then the contructor of this method needs to be called AFTER those definitions
# so they don't get overridden
class ConfigObject(PrintObject):

    def __init__(self, config):
        super().__init__()
        self.config = config
        if config is None:
            self.print_warning('ConfigObject received None as config')
            return
        self.print_ok('setting ' + config.firstChild.nodeValue + ' attributes')
        # load config parameters
        for key, value_text in config.attributes.items():
            try:
                value = eval(value_text)
            except NameError:
                value = value_text
            setattr(self, key, value)
            text = config.firstChild.nodeValue + '.' + key + '=' + value_text
            self.print_info(text)


# ----------


def print_error(*message):
    print('\033[91m', 'ERROR ', *message, '\033[0m')
    raise RuntimeError


def print_ok(*message):
    print('\033[92m', *message, '\033[0m')


def print_warning(*message):
    # yellow
    # print('\033[93m', *message, '\033[0m')
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


def wrap(val):
    ''' Wrap angle to [-pi,pi]
    '''
    return (val + np.pi) % (2*np.pi) - np.pi


def set_config_attr(config_minidom, config):
    """ Set the attributes in dom to config"""
    for key, value_text in config_minidom.attributes.items():
        if not hasattr(config, key):
            raise AttributeError(
                f'Config xml specified {key}={value_text},'
                f'but {key} does not exist in {config.__class__.__name__}')
        try:
            value = eval(value_text)
        except NameError:
            value = value_text
        setattr(config, key, value)
        return config
