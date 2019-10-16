cmake_minimum_required(VERSION 2.8.2)

include_directories(
    include
    ${ROS_LIB_DIR}
    ${catkin_INCLUDE_DIRS}
    
    )

#generate_arduino_firmware(arduino
#    SRCS arduino_main.cpp ${ROS_LIB_DIR}/time.cpp I2Cdev/I2Cdev.cpp MPU6050/MPU6050.cpp
#    BOARD nano328
#    PORT /dev/ttyUSB0
#)
#target_link_libraries(rc_vip_firmware_arduino ${catkin_LIBRARIES})
#add_dependencies(rc_vip_firmware_arduino  rc_vip_generate_messages_cpp)
