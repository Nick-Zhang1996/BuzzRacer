''' Terminate experiment after a number of time steps'''
from buzzracer.extension.Extension import Extension

# TODO merge this into LapCounter

class StepCounter(Extension):
    ''' Terminate experiment after a number of time steps'''
    def __init__(self):
        Extension.__init__(self,'step_counter')
        self.count = 0
        self.total_count = 20

    def update(self):
        self.count += 1
        if self.count >= self.total_count:
            self.main.exit_request.set()
