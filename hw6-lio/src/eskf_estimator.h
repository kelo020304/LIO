//
// Created by ubuntu on 24-2-26.
//

#ifndef DATA_GENERATOR_EKF_ESTIMATOR_H
#define DATA_GENERATOR_EKF_ESTIMATOR_H

#include "ros/ros.h"
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud.h>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>
#include "utility.h"
#include <queue>
#include <thread>
#include <mutex>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/kdtree/kdtree_flann.h>
#include "tic_toc.h"


enum StateIndex : uint {
    R = 0,                   // (3 dimention) rotation in world frame
    P = 3,                   // (3 dimention) position in world frame
    V = 6,                   // (3 dimention) velocity in world frame
    BA = 9,                 // (3 dimention) IMU acceleration bias
    BG = 12,                 // (3 dimention) IMU gyroscope bias
    STATE_TOTAL = 15
};

enum StateNoiseIndex : uint {
    ACC_NOISE = 0,         // linear acceleration change
    GYRO_NOISE = 3,          // angular velocity change
    ACC_RANDOM_WALK = 6,   // IMU aceleration bias random walk
    GYRO_RANDOM_WALK = 9,  // IMU gyroscope bias random walk
    NOISE_TOTAL = 12
};

struct State {
    Eigen::Quaterniond q;
    Eigen::Vector3d p;
    Eigen::Vector3d v;
    Eigen::Vector3d ba;
    Eigen::Vector3d bg;
    double time;
    Eigen::MatrixXd P;
};


class EskfEstimator {
public:
    EskfEstimator();
    ~EskfEstimator();
    void RosNodeRegistration(ros::NodeHandle &n);

private:
    void InitState(const double &time);
    void ImuCallback(const sensor_msgs::ImuConstPtr &imu_msg);
    void WheelCallback(const nav_msgs::OdometryConstPtr &wheel_msg);
    void LidarCallback(const sensor_msgs::PointCloudConstPtr &lidar_msg);
    void MainProcessThread();
    void PredictByImu(const Eigen::Vector3d &imuAcc, const Eigen::Vector3d &imuGyro, const double &dt);
    void UpdateByWheel(const double &time, const Eigen::Vector3d &wheelSpeed);
    void UpdateByLidar(const double &time, const std::vector<Eigen::Vector3d> &lidar_pts);
    void InitMap(const std::vector<Eigen::Vector3d> &lidar_pts);
    bool FindNearestPointInMap(const Eigen::Vector3d& local_pt, Eigen::Vector3d& map_pt, double threshold);
    void IncreasePointToMap(const std::vector<Eigen::Vector3d> &unmatched_local_pts);
    void PublishPose();
    void PublishPoints();

    ros::Subscriber sub_imu_, sub_wheel_, sub_lidar_;

    State state_;
    nav_msgs::Path path_;
    ros::Publisher pub_path_, pub_odometry_, pub_local_pts_, pub_global_pts_;

    Eigen::Vector3d gravity_;
    bool state_init_ = false;
    Eigen::MatrixXd Q_; // variance matrix of imu
    Eigen::MatrixXd Rm_; // variance matrix of wheel odometer
    Eigen::MatrixXd Rml_; // variance matrix of lidar

    bool shutdown_ = false;
    std::queue<sensor_msgs::ImuConstPtr> imu_buf_;
    std::queue<nav_msgs::OdometryConstPtr> odom_buf_;
    std::queue<sensor_msgs::PointCloudConstPtr> lidar_buf_;
    std::mutex m_buf_;
    std::thread main_thread_;

    std::vector<Eigen::Vector3d> map_cloud_;
    std::vector<Eigen::Vector3d> lidar_frame_points_;
    pcl::KdTreeFLANN<pcl::PointXYZ> map_kdtree_;
    bool map_init_ = false;

    double ACC_NOISE_VAR = 0.001;
    double GYRO_NOISE_VAR = 0.0001;
    double ACC_RANDOM_WALK_VAR = 0.0001;
    double GYRO_RANDOM_WALK_VAR = 0.00001;
    double WHEEL_ODOMETER_VAR = 0.1;
    double LIDAR_VAR = 0.001;
};


#endif //DATA_GENERATOR_EKF_ESTIMATOR_H
