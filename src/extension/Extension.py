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
