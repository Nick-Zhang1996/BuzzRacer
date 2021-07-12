
class Extension:
    def __init__(self,main):
        self.main = main

    # optional initialization
    def init(self):
        pass

    def preUpdate(self):
        pass

    def update(self):
        pass

    def postUpdate(self):
        pass

    def final(self):
        pass
