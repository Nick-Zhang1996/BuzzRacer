from common import *


class Extension(PrintObject):
    def __init__(self, main):
        Extension.main = main
        print_ok(self.prefix() + 'in use')
        main.extensions.append(self)
        self.name = self.__class__.__name__

    # optional initialization

    def pre_init(self):
        pass

    def init(self):
        pass

    def post_init(self):
        pass

    def pre_update(self):
        pass

    def update(self):
        pass

    def post_update(self):
        pass

    def pre_final(self):
        pass

    def final(self):
        pass

    def post_final(self):
        pass
