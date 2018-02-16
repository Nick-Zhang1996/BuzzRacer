#include <ros.h>
#include <std_msgs/Float64.h>

ros::NodeHandle nh;

std_msgs::Float64 throttle_val;
ros::Publisher throttle_pub("throttle_pub", &throttle_val);

float throttle = 1.0;

void setup() {
    nh.initNode();
    nh.advertise(throttle_pub);
}

void loop() {
    throttle_val.data = throttle;
    throttle_pub.publish(&throttle_val);
    nh.spinOnce();
    delay(1000);
}
