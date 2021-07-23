
class Extension:
    def __init__(self,main):
        Extension.main = main

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

    def final(self):
        pass
