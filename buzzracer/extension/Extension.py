''' Base class for extensions'''
from common import PrintObject, Config


class Extension(PrintObject):
    ''' Base class for extensions'''
    extensions = []

    def __init__(self, handle_name):
        super().__init__()
        self.name = handle_name
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
    def load(cls, main, config: Config):
        ''' Instantiate extensions as defined in config, set each extension 
        to an attribute of main with extension.name as attribute name,
        then set attributes of the extension as defined in config.'''
        Extension.main = main
        cls.print_ok('setting up extensions...')
        config_extensions = config.getElementsByTagName('extensions')[0]
        for config_extension in config_extensions.getElementsByTagName(
                'extension'):
            extension_class_name = config_extension.firstChild.nodeValue
            try:
                # pylint: disable-next=exec-used
                exec('from extension import ' + extension_class_name)
            except ImportError:
                cls.print_error(f'Cannot import {extension_class_name}')
                raise

            ext = eval(extension_class_name)()
            cls.print_info(f'Loading {extension_class_name}')
            setattr(main, ext.name, ext)
            for key, raw in config_extension.attributes.items():
                # try to parse the config as python statement, if fails
                # then interpret as string
                try:
                    value = eval(raw)
                except (NameError, SyntaxError):
                    value = raw
                # all other attributes in config will be added to extension
                setattr(ext, key, value)
                cls.print_info('main.' + ext.name + '.' + key + ' = ' +
                               str(value))

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
    def pre_update_all(cls):
        for extension in Extension.extensions:
            extension.pre_update()

    @classmethod
    def update_all(cls):
        for extension in Extension.extensions:
            extension.update()

    @classmethod
    def post_update_all(cls):
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
