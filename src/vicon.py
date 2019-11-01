#!/usr/bin/env python
# retrieve vicon feed from matlab and republish as ROS topic
import socket
import time
from struct import unpack


if False:
    IP = "0.0.0.0"
    PORT = 3883
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((IP, PORT))
    data, addr = sock.recvfrom(1024)
    print(data)
    newFile = open("samplevicon.bin", "wb")
    newFile.write(data)


f = open('samplevicon.bin','br')
data = f.read()

frameNumber = unpack('i',data[0:4])
itemsInBlock = data[4]
itemID = data[5]
itemDataSize = unpack('h',data[6:8])
itemName = data[8:32].decode("ascii")
# raw data in mm
x = unpack('d',data[32:40])
y = unpack('d',data[40:48])
z = unpack('d',data[48:56])
# euler angles, rotation order: rx,ry,rz, using intermediate frame
rx = unpack('d',data[56:64])
ry = unpack('d',data[64:72])
rz = unpack('d',data[72:80])
print(x,y,z,rx,ry,rz)
