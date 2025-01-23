from extension.Extension import Extension
from track.RCPTrack import RCPTrack
import numpy as np

class TofSensorVisualization(Extension):
    def __init__(self,main):
        Extension.__init__(self,main)

        self.car_id = [0]
        self.scale = 1/1000
        self.max_range = 1000

    def init(self):
        track = self.main.track
        assert(isinstance(track,RCPTrack))
        
        inf = float('inf')
        for i in self.car_id:
            self.main.cars[i].tof_measurement = np.array([self.max_range,self.max_range,self.max_range,self.max_range], dtype=np.float64)

    def preUpdate(self):
        # car.tof_measurement = [front, left, right, rear], in meter
        for i in self.car_id:
            car = self.main.cars[i]
            self.plotTof(car)


    def plotTof(self,car):
        if (self.main.visualization.update_visualization.is_set()):
            img = self.main.visualization.visualization_img
            # front
            x = car.states[0] + car.lidar_front * np.cos(car.states[2])
            y = car.states[1] + car.lidar_front * np.sin(car.states[2])
            self.main.track.drawCircle(img, (x,y), 0.05)
            # left
            x = car.states[0] + car.lidar_left * np.cos(car.states[2] + np.pi/2)
            y = car.states[1] + car.lidar_left * np.sin(car.states[2] + np.pi/2)
            self.main.track.drawCircle(img, (x,y), 0.05)
            # right
            x = car.states[0] + car.lidar_right * np.cos(car.states[2] - np.pi/2)
            y = car.states[1] + car.lidar_right * np.sin(car.states[2] - np.pi/2)
            self.main.track.drawCircle(img, (x,y), 0.05)
            # rear
            x = car.states[0] + car.lidar_rear * np.cos(car.states[2] + np.pi)
            y = car.states[1] + car.lidar_rear * np.sin(car.states[2] + np.pi)
            self.main.track.drawCircle(img, (x,y), 0.05)
            self.main.visualization.visualization_img = img
