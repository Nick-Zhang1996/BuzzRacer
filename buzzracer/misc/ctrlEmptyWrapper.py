# empty controller, does nothing

class ctrlEmptyWrapper(Car):
    def __init__(self):
        return
    def init(self, track):
        return
    def ctrlCar(self,states,track,v_override=None,reverse=False):
        return (0,0)
