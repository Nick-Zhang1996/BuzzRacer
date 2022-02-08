from common import *
class Extension:
    def __init__(self,main):
        Extension.main = main
        print_ok(self.prefix() + "in use")
        main.extensions.append(self)

    def prefix(self):
        return "["+self.__class__.__name__+"]: "

    # optional initialization
    def init(self):
        pass

    def postInit(self):
        pass

    def preUpdate(self):
        pass

    def update(self):
        pass

    def postUpdate(self):
        pass

    def preFinal(self):
        pass
    def final(self):
        pass
    def postFinal(self):
        pass

    def print_error(self,*message):
        print('\033[91m',self.prefix(), 'ERROR ', *message, '\033[0m')
        raise RuntimeError
    def print_ok(self,*message):
        print('\033[92m',self.prefix(), *message, '\033[0m')

    def print_warning(self,*message):
        # yellow
        #print('\033[93m', *message, '\033[0m')
        # red
        print('\033[91m',self.prefix(), 'WARNING: ', *message, '\033[0m')

    def print_info(self,*message):
        print('\033[96m',self.prefix(), *message, '\033[0m')
