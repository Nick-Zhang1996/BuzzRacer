# test script for dynamically configure ros parameter

import rospy
from time import sleep

rospy.set_param('setted',10.1)

while True:
    print(rospy.get_param('setted'))
    sleep(1)
