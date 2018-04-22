import cv2
import pickle
import numpy as np
from os import listdir
from os.path import isfile, join

class imageutil:

    # provide path to the folder that contains mtx.p and dist.p
    # if no path is given, user is expected to call calibration() and establish new matrixes
    def __init__(self, path_to_file = None):
        self.mtx = None
        self.dist = None

        if path_to_file is None:
            # User will call calibration to calibrate
            return
        else:
            #read in calibrated file
            with open(path_to_file+'mtx.p', 'rb') as f:
                 self.mtx = pickle.load(f)
            with open(path_to_file+'dist.p', 'rb') as f:
                 self.dist = pickle.load(f)
            #XXX need to check failure cases
            print('matrixes loaded')
            return

    def calibration(self,path_to_file,path_to_output):
        print('calibrating....')
        filenames = [f for f in listdir(path_to_file) if isfile(join(path_to_file, f))]
        objpoints_set=[]
        imgpoints_set=[]
        # Use 9*6 chessboard
        objpoints = np.zeros([9*6,3],np.float32)
        objpoints[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
        
        for filename in filenames:
            image = cv2.imread(path_to_file+filename)
            ret, corners = cv2.findChessboardCorners(image, (9,6))
            if ret:
                #imgpoints_set.append(np.squeeze(corners))
                imgpoints_set.append(corners)
                objpoints_set.append(objpoints)

        ret, mtx, dist, rvecs,tvecs = cv2.calibrateCamera(objpoints_set,imgpoints_set,image.shape[0:2],None, None)
                
        with open(path_to_output+'mtx.p', 'wb') as f:
                pickle.dump(mtx, f, pickle.HIGHEST_PROTOCOL)
        with open(path_to_output+'dist.p', 'wb') as f:
                pickle.dump(dist, f, pickle.HIGHEST_PROTOCOL)
        self.mtx = mtx
        self.dist = dist
        print('calibration completed')

        return mtx, dist

    def undistort(self,img):
        dst = cv2.undistort(img,self.mtx,self.dist)
        return dst

    def undistortPts(self,src):
        dst = cv2.undistortPoints(src,self.mtx,self.dist,P=self.mtx)
        return dst


if __name__ == '__main__':
    folder = "../"
    t = imageutil(None)
    t.calibration(folder+"calibration/",folder+"calibrated/")
    path_to_file = folder+"calibration/"
    filenames = [f for f in listdir(path_to_file) if isfile(join(path_to_file, f))]
    filename = path_to_file +filenames[0]
    import matplotlib.pyplot as plt
    image = cv2.imread(filename)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    plt.show()
    image = t.undistort(image)
    print(t.undistortPts(np.array([[[95,394],[104,70]]],dtype=np.float32)))
    plt.imshow(image)
    plt.show()
    
