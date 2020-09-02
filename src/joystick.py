# joystick/manual controller
# use left konb for steering
# right trigger for throttle
# left trigger for braking
from inputs import get_gamepad
from threading import Thread,Lock,Event
from common import *
from time import sleep

class Joystick:
    def __init__(self,):
        self.gamepad = {}
        self.exit_signal = Event()

        self.used_channels = ['ABS_X','ABS_Z','ABS_RZ']
        print_info("please move all joystick to initialize")
        while not (all([x in self.gamepad for x in self.used_channels])) and not self.exit_signal.isSet():
            self.update()
            sleep(0.01)
        print_ok("Success")

        self.child_threads = []
        self.child_threads.append(Thread(name='joystick', target=self.updateDaemon))
        self.child_threads[-1].start()

    def quit(self):
        self.exit_signal.set()
        for x in self.child_threads:
            x.join()
        
    # read gamepad, update command
    def update(self,):
        events = get_gamepad()
        for e in events:
            if e.ev_type == 'Absolute':
                self.gamepad[e.code] = e.state

    def updateDaemon(self,):
        while not self.exit_signal.isSet():
            self.update()
            # left:1, right:-1
            self.steering = -np.clip(float(self.gamepad['ABS_X']) / 32767.0, -1, 1)
            # full reverse/brake: -1, full throttle:1
            self.throttle = np.clip(float(self.gamepad['ABS_RZ']) / 1024.0 - float(self.gamepad['ABS_Z']) / 1024.0, -1, 1)*30.0

if __name__ == '__main__':
    joy = Joystick()

