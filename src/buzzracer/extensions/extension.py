''' Base class for extensions'''
from __future__ import annotations
from typing import TYPE_CHECKING
import logging

from buzzracer.common import PrintObject, Config, set_config_attr
from buzzracer.utilities.execution_timer import ExecutionTimer

if TYPE_CHECKING:
    from buzzracer.main import MainConfig

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ExtensionConfig:
    """ Base class for Extension Config.
    Subclasses can inherit this class and register the subclass in Extension.register"""

    def __init__(self, main_config):
        del main_config
        self.name = ''
        """ Name of this extension module (e.g. visualization, simulator)"""


class ExtensionState:
    """ Base class for Extension State.
    Subclasses can inherit this class and register the subclass in Extension.register"""

    def __init__(self, config: ExtensionState):
        pass


class Extension(PrintObject):
    ''' Base class for extensions'''
    registry = {}
    config_registry = {}
    state_registry = {}
    handle_name_registry = {}

    extensions = []

    def __init__(self):
        super().__init__()
        Extension.extensions.append(self)

    # optional initialization
    def pre_init(self):
        ''' Initialization to run before normal initializations. '''

    def init(self):
        ''' Initializations for this extension'''

    def post_init(self):
        ''' Initialization to run after normal initializations. '''

    def pre_update(self):
        ''' Update to run before normal updates'''

    def update(self):
        ''' Update function to be called at each iteration'''

    def post_update(self):
        ''' Update to run after normal updates'''

    def pre_final(self):
        pass

    def final(self):
        pass

    def post_final(self):
        pass

    @classmethod
    def load(cls, main, main_config: MainConfig, config_minidom: Config):
        ''' Instantiate extensions as defined in config, set each extension 
        to an attribute of main with extension.config.name as attribute name,
        then set attributes of the extension as defined in config.'''
        Extension.main = main
        cls.print_ok('setting up extensions...')
        for config_extension in config_minidom.getElementsByTagName('extension'):
            name = config_extension.firstChild.nodeValue  # Extension class name
            obj, obj_config, _ = Extension.factory(name, main_config, config_minidom)
            # e.g. main.visualization = Visualization
            # The name of an object is set at registration, it is set as the attribute of Main
            setattr(main, obj_config.name, obj)

    @staticmethod
    def register(handle_name, config_cls, state_cls):
        """Decorator to add a controller class to the registry."""
        def wrapper(cls):
            name = cls.__name__
            Extension.registry[name] = cls
            Extension.config_registry[name] = config_cls
            Extension.state_registry[name] = state_cls
            Extension.handle_name_registry[name] = handle_name
            print(f'registered {name}')
            return cls
        return wrapper

    @staticmethod
    def factory(name, main_config, config_minidom):
        obj_cls = Extension.registry[name]
        config_cls = Extension.config_registry[name]
        state_cls = Extension.state_registry[name]
        handle_name = Extension.handle_name_registry[name]
        config = config_cls(main_config)
        config = set_config_attr(config_minidom, config)
        state = state_cls(config)
        obj = obj_cls()
        obj.config = config
        obj.state = state
        setattr(config, 'name', handle_name)
        return (obj, config, state)

    @classmethod
    def pre_init_all(cls):
        for extension in Extension.extensions:
            extension.pre_init()

    @classmethod
    def init_all(cls):
        for extension in Extension.extensions:
            extension.init()

    @classmethod
    def post_init_all(cls):
        for extension in Extension.extensions:
            extension.post_init()

    @classmethod
    def pre_update_all(cls, t: ExecutionTimer = None):
        if isinstance(t, ExecutionTimer):
            for extension in Extension.extensions:
                t.s(extension.config.name)
                extension.pre_update()
                t.e(extension.config.name)
        else:
            for extension in Extension.extensions:
                extension.pre_update()

    @classmethod
    def update_all(cls, t: ExecutionTimer = None):
        if isinstance(t, ExecutionTimer):
            for extension in Extension.extensions:
                t.s(extension.config.name)
                extension.update()
                t.e(extension.config.name)
        else:
            for extension in Extension.extensions:
                extension.update()

    @classmethod
    def post_update_all(cls, t: ExecutionTimer = None):
        if isinstance(t, ExecutionTimer):
            for extension in Extension.extensions:
                t.s(extension.config.name)
                extension.post_update()
                t.e(extension.config.name)
        else:
            for extension in Extension.extensions:
                extension.post_update()

    @classmethod
    def pre_final_all(cls):
        for extension in Extension.extensions:
            extension.pre_final()

    @classmethod
    def final_all(cls):
        for extension in Extension.extensions:
            extension.final()

    @classmethod
    def post_final_all(cls):
        for extension in Extension.extensions:
            extension.post_final()
