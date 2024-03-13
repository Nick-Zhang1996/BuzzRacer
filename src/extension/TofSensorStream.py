from extension.Extension import Extension
from track.RCPTrack import RCPTrack
import numpy as np

class TofSensorStream(Extension):
    def __init__(self,main):
        Extension.__init__(self,main)

        self.car_id = [0]
        self.scale = 1/1000

    def init(self):
        track = self.main.track
        assert(isinstance(track,RCPTrack))
        
        inf = float('inf')
        for i in self.car_id:
            self.main.cars[i].tof_measurement = (0.2,0.2,0.2,0.2)

    def preUpdate(self):
        # car.tof_measurement = [front, left, right, rear], in meter
        for i in self.car_id:
            car = self.main.cars[i]
            last_packet = car.last_packet

            if last_packet != None:
                measurement = [last_packet.lidar_front, last_packet.lidar_left, last_packet.lidar_right, last_packet.lidar_back]

                print(measurement)

                for i, m in enumerate(measurement):
                    if m <= 0:
                        measurement[i] = 0
                
                measurement = np.array(measurement) * self.scale

                car.tof_measurement = measurement

            self.plotTof(car)


    def plotTof(self,car):
        if (self.main.visualization.update_visualization.is_set()):
            img = self.main.visualization.visualization_img
            # front
            if (not car.tof_measurement[0] is None):
                x = car.states[0] + car.tof_measurement[0] * np.cos(car.states[2])
                y = car.states[1] + car.tof_measurement[0] * np.sin(car.states[2])
                self.main.track.drawCircle(img, (x,y), 0.05)
            # left
            if (not car.tof_measurement[1] is None):
                x = car.states[0] + car.tof_measurement[1] * np.cos(car.states[2] + np.pi/2)
                y = car.states[1] + car.tof_measurement[1] * np.sin(car.states[2] + np.pi/2)
                self.main.track.drawCircle(img, (x,y), 0.05)
            # right
            if (not car.tof_measurement[2] is None):
                x = car.states[0] + car.tof_measurement[2] * np.cos(car.states[2] - np.pi/2)
                y = car.states[1] + car.tof_measurement[2] * np.sin(car.states[2] - np.pi/2)
                self.main.track.drawCircle(img, (x,y), 0.05)
            # rear
            if (not car.tof_measurement[3] is None):
                x = car.states[0] + car.tof_measurement[3] * np.cos(car.states[2] + np.pi)
                y = car.states[1] + car.tof_measurement[3] * np.sin(car.states[2] + np.pi)
                self.main.track.drawCircle(img, (x,y), 0.05)
            self.main.visualization.visualization_img = img