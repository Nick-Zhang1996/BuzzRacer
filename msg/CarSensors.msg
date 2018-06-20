# This message holds values for all sensors onboard rc-vip vehicle
# The arduino would read from available sensors and publish to \sensors
# However, not every field is filled, and it is up to the maintainer to keep track of which ones are actually utilized. If you implement/abandon some fields described in this message, please update in the comments accordingly.
# This message includes other compound messages types provided by ROS, for details on how to use each field (units, meaning, etc) search for original .msg files in associated ros packages

std_msgs/Header header
sensor_msgs/BatteryState batt# voltage and current are published
sensor_msgs/Imu imu

float32 driveshaft_speed # rotation speed of the driveshaft connecting front and rear axle. unit: rev/sec, this should be a smoothed value that requires no additional handling on the subscriber's part

