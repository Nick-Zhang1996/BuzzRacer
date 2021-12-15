# trying to recreate issue

import matplotlib
import matplotlib.pyplot as plt

import cv2
from PIL import Image
from RCPTrack import RCPtrack
from time import sleep
import numpy as np

stateArr = np.load("/home/caleb/Documents/GitHub/RC-VIP/src/stateValues.npy")
controlArr = np.load("/home/caleb/Documents/GitHub/RC-VIP/src/controlValues.npy")

xActual = stateArr[:-1, 0]
yActual = stateArr[:-1, 1]
headingActual = stateArr[:-1, 2]
vxActual = stateArr[:-1, 3]
vyActual = stateArr[:-1, 4]
omegaActual = stateArr[:-1, 5]

throttle = controlArr[:-1, 0]
steering = controlArr[:-1, 1]