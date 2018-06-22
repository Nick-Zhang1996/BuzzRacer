cmake_minimum_required(VERSION 2.8.2)

include_directories(
    include
    ${ROS_LIB_DIR}
    ${catkin_INCLUDE_DIRS}
    
    )

generate_arduino_firmware(arduino
    SRCS arduino_main.cpp ${ROS_LIB_DIR}/time.cpp
    BOARD nano328
    PORT /dev/ttyUSB0
)
