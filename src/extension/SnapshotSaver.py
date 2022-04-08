from extension.Extension import Extension
import matplotlib.pyplot as plt
import cv2
class SnapshotSaver(Extension):
    def __init__(self,main):
        Extension.__init__(self,main)

    def postInit(self):
        self.background = self.main.track.drawTrack()
        self.background = self.main.track.drawRaceline(img=self.background)

        self.img = self.background.copy()
        self.timestep = 0
        self.snapshot_count = 0
        self.start = -1
        self.end = -1
        self.interval = 20

    def takeSnapshot(self,duration=100,interval=20):
        self.start = self.timestep
        self.end = self.timestep+duration
        self.interval = interval

    def postUpdate(self):
        self.timestep +=1
        img = self.img
        # save a multiple exposure photo
        if (self.timestep >= self.start and self.timestep <= self.end and self.timestep%self.interval == 0):
            for car in self.main.cars:
                img = self.main.visualization.drawCar(img, car)
                self.print_info("draw car at frame %d"%(self.timestep))
        self.img = img

        if (self.timestep > self.end and self.end>0):
            self.snapshot_count += 1
            self.end = -1
            filename = "./snapshot%d.png"%(self.snapshot_count)
            cv2.imwrite(filename,self.img)
            self.print_info(self.prefix()+"saved snapshot at "+filename)
            plt.imshow(cv2.cvtColor(self.img,cv2.COLOR_BGR2RGB))
            plt.show()
            self.img = self.background.copy()

