from extension.Extension import Extension
class StepCounter(Extension):
    def __init__(self,main):
        Extension.__init__(self,main)
        self.count = 0
        self.total_count = 20
    def update(self):
        self.count += 1
        if (self.count >= self.total_count):
            self.main.exit_request.set()
