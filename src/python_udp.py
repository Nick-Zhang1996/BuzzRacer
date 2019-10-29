#!/usr/bin/env python
# retrieve vicon feed from matlab and republish as ROS topic
import rospy
import socket
import time
from rcvip_msgs.msg import Vicon

def vicon(pub_vicon):
	#IP = '192.168.7.11'
	#IP = '0.0.0.0'
        IP = "0.0.0.0"
	PORT = 3883
	sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
	sock.bind((IP, PORT))
	data, addr = sock.recvfrom(1024)
	data = data.strip('[]')
	data = data.split(';')
	pub_data = []
	for d in data:
		pub_data.append(float(d))
	#print("received message:", pub_data)
	pub_vicon.publish(pub_data)
	#time.sleep(0.1)

if __name__ == '__main__':
	rospy.init_node("vicon_translator")
	pub_vicon = rospy.Publisher("/vicon_tf", Vicon, queue_size=1)
	while not rospy.is_shutdown():	
		vicon(pub_vicon)
