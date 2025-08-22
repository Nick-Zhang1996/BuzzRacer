# TODO figure out dimension/shape
# read video frames
import pickle
import numpy as np
import cv2 as cv


class Video:
    def __init__(self):
        # time in video that corresponds to start of log
        self.time_offset = 10.45
        self.fps = 30
        # load transformation
        with open('resources/transform.p', 'rb') as f:
            self.mat = pickle.load(f)

        # load state
        # time, x, y, heading, vx, vy, omega, throttle, steering
        # timestep, car_id, data_index
        with open('resources/full_state3.p', 'rb') as f:
            self.log = np.array(pickle.load(f))
        self.log[:, :, 0] -= self.log[0, 0, 0]

        self.cap = cv.VideoCapture('resources/IMG_6754.MOV')
        fourcc = cv.VideoWriter_fourcc(*'XVID')
        self.out = cv.VideoWriter(
            'outputs/output.avi', fourcc, 30.0, (1920, 1080))
        if not self.cap.isOpened():
            print('output error')
            exit()
        if not self.cap.isOpened():
            print('Cannot open camera')
            exit()

    def getStateAtTime(self, t):
        # TODO add interpolation
        try:
            index = np.searchsorted(self.log[:, 0, 0], t)
            return self.log[index]
        except IndexError:
            return self.log[-1]

    def render(self, frame, t):
        # only for car 0
        log = self.getStateAtTime(t)
        car_pos = log[0, 1:3].reshape(1, 1, -1).astype(np.float32)
        dst = cv.perspectiveTransform(car_pos, self.mat)
        dst = tuple(dst.flatten().astype(int))
        cv.circle(frame, dst, 55, (0, 0, 255), 3)
        return frame

    def drawPolyline(self, points, frame):
        # TODO aike
        return

    def drawCircle(self, center, radius, frame):
        # TODO aike
        return

    def drawText(self, bottom_left, text, frame):
        # TODO aike
        return

    def main(self):
        cap = self.cap
        video_t = 0.0 - 1/self.fps
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            video_t += 1/self.fps
            if (video_t < 10):
                continue
            print(f'video_t: {video_t}')
            # if frame is read correctly ret is True
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            frame = self.render(frame, video_t-self.time_offset)
            self.out.write(frame)

            # Display the resulting frame
            cv.imshow('frame', frame)
            if cv.waitKey(1) == ord('q'):
                break
        # When everything done, release the capture
        cap.release()
        self.out.release()
        cv.destroyAllWindows()


if __name__ == '__main__':
    main = Video()
    main.main()
