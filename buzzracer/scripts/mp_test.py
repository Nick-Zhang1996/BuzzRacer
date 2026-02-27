""" Multiprocess example """
import ctypes
from time import sleep
from multiprocessing import Process, Value, Lock, Event


class CarState(ctypes.Structure):
    _fields_ = [
        ('x', ctypes.c_float),
        ('y', ctypes.c_float),
        ('heading', ctypes.c_float),
    ]

    def to_dict(self):
        return {field_name: getattr(self, field_name) for field_name, _ in self._fields_}


class Main:
    def __init__(self):
        self.child_processes = []
        self.sv_car_states = []
        self.e_new_state = []
        self.e_exit = Event()
        for i in range(10):
            sv_car_state = Value(CarState, 0, 1.1, 2.2)
            e_new_state = Event()
            p = Process(target=self.controller_process_fun,
                        args=(i, sv_car_state, e_new_state, self.e_exit))
            p.start()
            self.child_processes.append(p)
            self.sv_car_states.append(sv_car_state)
            self.e_new_state.append(e_new_state)

    @staticmethod
    def controller_process_fun(i, sv_car_state, e_new_state, e_exit):
        while not e_exit.wait(0.01):
            if e_new_state.wait(0.01):
                print(
                    f'running control for car {i}, {sv_car_state.get_obj().to_dict()}')
                e_new_state.clear()

    def run(self):
        try:
            for k in range(100):
                for i in range(10):
                    self.sv_car_states[i].x = k*0.1
                    self.sv_car_states[i].y = k*0.2
                    self.e_new_state[i].set()
                sleep(0.01)
            self.e_exit.set()
        finally:
            for p in self.child_processes:
                p.join()


if __name__ == '__main__':
    main = Main()
    main.run()
