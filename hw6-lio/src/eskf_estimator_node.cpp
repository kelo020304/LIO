//
// Created by ubuntu on 24-2-26.
//
#include "ros/ros.h"
#include "eskf_estimator.h"

using namespace std;


int main(int argc, char **argv) {
    ros::init(argc, argv, "eskf_estimator");
    ros::NodeHandle n("~");

    std::cout<<"111  "<< std::endl;
    EskfEstimator estimator;
    estimator.RosNodeRegistration(n);

    ros::spin();
}
