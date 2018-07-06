# This message holds values for all sensors onboard rc-vip vehicle
# The arduino would read from available sensors and publish to \sensors
# However, not every field is filled, and it is up to the maintainer to keep track of which ones are actually utilized. If you implement/abandon some fields described in this message, please update in the comments accordingly.
# This message includes other compound messages types provided by ROS, for details on how to use each field (units, meaning, etc) search for original .msg files in associated ros packages

# Normally we should use official ROS msg type whenever applicable, 
# However, these messages are too bulky in size for the Atmel 328p to handle with its
# 2kB RAM
#
std_msgs/Header header
#sensor_msgs/BatteryState batt# voltage and current are published

#battery voltage
float32 voltage

#Not using ROS's imu type cuz it's full of float64 and depletes memory on Arduino
#sensor_msgs/Imu imu
#linear acceleration
int16 imu_ax
int16 imu_ay
int16 imu_az

# angular speed
int16 imu_gx
int16 imu_gy
int16 imu_gz

#float32 driveshaft_speed # rotation speed of the driveshaft connecting front and rear axle. unit: rev/sec, this should be a smoothed value that requires no additional handling on the subscriber's part

