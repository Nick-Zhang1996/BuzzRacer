from extension.Extension import Extension
import matplotlib.pyplot as plt
import cv2
from threading import Event
class SnapshotSaver(Extension):
    def __init__(self,main):
        Extension.__init__(self,main)
        self.recording = Event()

    def postInit(self):
        self.background = self.main.track.drawTrack()
        self.background = self.main.track.drawRaceline(img=self.background)

        self.img = None
        self.timestep = 0
        self.snapshot_count = 0
        self.interval = 20

    # this will be called when user press 's'
    # first time this will start snapshot, second time will stop
    def takeSnapshot(self):
        if (self.recording.is_set()):
            self.recording.clear()
            self.print_info('snapshot stopping')
        else:
            self.recording.set()
            self.timestep = 0
            self.print_info('snapshot started')


    def postUpdate(self):
        self.timestep +=1
        # save a multiple exposure photo
        if (self.recording.is_set() and self.timestep%self.interval == 0):
            if (self.img is None):
                img = self.background.copy()
            else:
                img = self.img
            for car in self.main.cars:
                img = self.main.visualization.drawCar(img, car)
                self.print_info("snapshot at frame %d"%(self.timestep))
            self.img = img

        if (not self.recording.is_set() and not (self.img is None)):
            self.snapshot_count += 1
            filename = "./snapshot%d.png"%(self.snapshot_count)
            cv2.imwrite(filename,self.img)
            self.print_info(self.prefix()+"saved snapshot at "+filename)
            plt.imshow(cv2.cvtColor(self.img,cv2.COLOR_BGR2RGB))
            plt.show()
            self.img = None

