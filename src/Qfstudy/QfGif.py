import numpy as np
import glob
from PIL import Image
import cv2
gif_filename = "qf.gif"
imgs = []
#for path in glob.glob("./*.png"):

count = 20
for i in np.linspace(0,3,31):
    count += 1
    if count < 0:
        break
    path = "./ccmppi_Qf_"+str(i)+".png"
    img = cv2.imread(path,cv2.IMREAD_COLOR)
    print("+1")
    #imgs.append(Image.fromarray(img))
    imgs.append(Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB)))


#imgs.append(Image.fromarray(cv2.cvtColor(self.main.visualization.visualization_img.copy(),cv2.COLOR_BGR2RGB)))
imgs[0].save(fp=gif_filename,format='GIF',append_images=imgs,save_all=True,duration = 150,loop=0)
