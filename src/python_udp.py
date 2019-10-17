#!/usr/bin/env python
import rospy
import socket
import time
from rc_vip.msg import Vicon


def vicon(pub_vicon):
	IP = ''
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
	time.sleep(0.1)

if __name__ == '__main__':
	rospy.init_node("udp_listener")
	pub_vicon = rospy.Publisher("/vicon_tf", Vicon, queue_size=10)
	while not rospy.is_shutdown():	
		vicon(pub_vicon)
