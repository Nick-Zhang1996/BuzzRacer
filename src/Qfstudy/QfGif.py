import glob
from PIL import Image
import cv2
gif_filename = "qf.gif"
imgs = []
#for path in glob.glob("./*.png"):

for i in range(51):
    path = "./ccmppi_Qf_"+str(i)+".png"
    img = cv2.imread(path,cv2.IMREAD_COLOR)
    imgs.append(Image.fromarray(img))


#imgs.append(Image.fromarray(cv2.cvtColor(self.main.visualization.visualization_img.copy(),cv2.COLOR_BGR2RGB)))
imgs[0].save(fp=gif_filename,format='GIF',append_images=imgs,save_all=True,duration = 200,loop=0)
