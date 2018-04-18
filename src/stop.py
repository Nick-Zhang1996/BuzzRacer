import rospy
from std_msgs.msg import Float64 as float_msg

def talker():
    throttle_pub = rospy.Publisher("/throttle",float_msg, queue_size=1)
    steering_pub = rospy.Publisher("/steer_angle",float_msg, queue_size=1)
    rospy.init_node('publisher', anonymous=True)
    rate = rospy.Rate(10) # 10hz
    while not rospy.is_shutdown():
        throttle_pub.publish(0)
        steering_pub.publish(0)
        rate.sleep()

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
