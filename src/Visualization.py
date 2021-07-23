import cv2
from time import sleep,time
from common import *
from Extension import Extension
class Visualization(Extension):
    def __init__(self,main):
        super().__init__(main)


    def final(self):
        cv2.destroyAllWindows()

    def init(self,):
        self.visualization_ts = time()
        self.img_track = self.main.track.drawTrack()
        self.img_track = self.main.track.drawRaceline(img=self.img_track)
        cv2.imshow('experiment',self.img_track)
        cv2.waitKey(1)

    def update(self,):
        # restrict update rate to 0.02s/frame, a rate higher than this can lead to frozen frames
        if (time()-self.visualization_ts>0.02):
            img = self.img_track.copy()
            for car in self.main.cars:
                img = self.main.track.drawCar(img, car.states, car.steering)
                # FIXME
                continue

                # CCMPPI
                if (car.controller == Controller.ccmppi):

                    # plot sampled trajectory (if car follow one sampled control traj)
                    coords_vec = self.debug_dict[car.id]['rollout_traj_vec']
                    for coords in coords_vec:
                        img = self.main.track.drawPolyline(coords,lineColor=(200,200,200),img=img)

                    # plot ideal trajectory (if car follow synthesized control)
                    coords = self.debug_dict[car.id]['ideal_traj']
                    for coord in coords:
                        x,y = coord
                        img = self.main.track.drawPoint(img,(x,y),color=(255,0,0))
                    img = self.main.track.drawPolyline(coords,lineColor=(100,0,100),img=img)

                    # plot opponent prediction
                    '''
                    coords_vec = self.debug_dict[car.id]['opponent_prediction']
                    for coords in coords_vec:
                        for coord in coords:
                            x,y = coord
                            img = self.main.track.drawPoint(img,(x,y),color=(255,0,0))
                        img = self.main.track.drawPolyline(coords,lineColor=(100,0,0),img=img)
                    '''

                    '''
                    coords_vec = np.array(coords_vec)
                    for i in range(len(coords_vec)):
                        plt.plot(coords_vec[0,:,0], coords_vec[0,:,1])
                    plt.show()
                    '''

            # TODO 
            '''
            if 'opponent' in self.debug_dict[0]:
                x_ref = self.debug_dict[0]['opponent']
                for coord in x_ref[0]:
                    x,y = coord
                    img = self.main.track.drawPoint(img,(x,y),color=(255,0,0))
            '''

            # plot reference trajectory following some alternative control sequence
            '''
            x_ref_alt = self.debug_dict[0]['x_ref_alt']
            for samples in x_ref_alt:
                for coord in samples:
                    x,y = coord
                    img = self.main.track.drawPoint(img,(x,y),color=(100,0,0))
            '''

            self.visualization_ts = time()
            self.visualization_img = img
            cv2.imshow('experiment',img)

            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                # first time q is presed, slow down
                if not self.main.slowdown.isSet():
                    print_ok("slowing down, press q again to shutdown")
                    self.main.slowdown.set()
                    self.main.slowdown_ts = time()
                else:
                    # second time, shut down
                    self.main.exit_request.set()
