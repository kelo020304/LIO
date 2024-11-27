//
// Created by ubuntu on 24-2-26.
//

#include "eskf_estimator.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/flann/dist.h>
#include <pcl/registration/icp.h>

EskfEstimator::EskfEstimator() {
    std::cout << "2222" << std::endl;
    main_thread_ = std::thread(&EskfEstimator::MainProcessThread, this);
}

EskfEstimator::~EskfEstimator() {
    std::cout << " sb " << std::endl;
    shutdown_ = true;
    main_thread_.join();
}

void EskfEstimator::InitState(const double &time) {
    ROS_WARN("init state");
    state_.q = Eigen::Quaterniond(1, 0, 0, 0);
    state_.p = Eigen::Vector3d(0, 0, 0);
    state_.v = Eigen::Vector3d(0, 0, 0);
    state_.ba = Eigen::Vector3d(0, 0, 0);
    state_.bg = Eigen::Vector3d(0, 0, 0);
    state_.P = Eigen::Matrix<double, StateIndex::STATE_TOTAL, StateIndex::STATE_TOTAL>::Identity();
    state_.time = time;
    gravity_ = Eigen::Vector3d(0, 0, -9.805);
    state_init_ = true;
    Q_ = Eigen::Matrix<double, StateNoiseIndex::NOISE_TOTAL, StateNoiseIndex::NOISE_TOTAL>::Identity();
    Q_.block<3, 3>(StateNoiseIndex::ACC_NOISE, StateNoiseIndex::ACC_NOISE) =
            ACC_NOISE_VAR * Eigen::Matrix3d::Identity();
    Q_.block<3, 3>(StateNoiseIndex::GYRO_NOISE, StateNoiseIndex::GYRO_NOISE) =
            GYRO_NOISE_VAR * Eigen::Matrix3d::Identity();
    Q_.block<3, 3>(StateNoiseIndex::ACC_RANDOM_WALK, StateNoiseIndex::ACC_RANDOM_WALK) =
            ACC_RANDOM_WALK_VAR * Eigen::Matrix3d::Identity();
    Q_.block<3, 3>(StateNoiseIndex::GYRO_RANDOM_WALK, StateNoiseIndex::GYRO_RANDOM_WALK) =
            GYRO_RANDOM_WALK_VAR * Eigen::Matrix3d::Identity();
    Rm_ = Eigen::Matrix3d::Identity() * WHEEL_ODOMETER_VAR;
    Rml_ = Eigen::Matrix3d::Identity() * LIDAR_VAR;
    ROS_WARN("init state finish");
}

void EskfEstimator::ImuCallback(const sensor_msgs::ImuConstPtr &imu_msg) {
    m_buf_.lock();
    imu_buf_.emplace(imu_msg);
    while (imu_buf_.size() > 1000) {
        imu_buf_.pop();
        ROS_WARN("throw imu measurement! %lf", imu_msg->header.stamp.toSec());
    }
    m_buf_.unlock();
}

void EskfEstimator::WheelCallback(const nav_msgs::OdometryConstPtr &wheel_msg) {
    m_buf_.lock();
    odom_buf_.emplace(wheel_msg);
    while (odom_buf_.size() > 1000) {
        odom_buf_.pop();
        ROS_WARN("throw wheel measurement! %lf", wheel_msg->header.stamp.toSec());
    }
    m_buf_.unlock();
}


void EskfEstimator::LidarCallback(const sensor_msgs::PointCloudConstPtr &lidar_msg) {
    m_buf_.lock();
    lidar_buf_.emplace(lidar_msg);
    while (lidar_buf_.size() > 1000) {
        lidar_buf_.pop();
        ROS_WARN("throw lidar measurement! %lf", lidar_msg->header.stamp.toSec());
    }
    m_buf_.unlock();
}

void EskfEstimator::RosNodeRegistration(ros::NodeHandle &n) {
    sub_imu_ = n.subscribe("/data_generator/imu", 2000, &EskfEstimator::ImuCallback, this,
                           ros::TransportHints().tcpNoDelay());
    sub_wheel_ = n.subscribe("/data_generator/odometry", 2000, &EskfEstimator::WheelCallback, this,
                             ros::TransportHints().tcpNoDelay());
    sub_lidar_ = n.subscribe("/data_generator/lidar_cloud", 2000, &EskfEstimator::LidarCallback, this,
                             ros::TransportHints().tcpNoDelay());
    pub_path_ = n.advertise<nav_msgs::Path>("path", 1000);
    pub_odometry_ = n.advertise<nav_msgs::Odometry>("odometry", 1000);
    path_.header.frame_id = "world";

    pub_local_pts_ = n.advertise<sensor_msgs::PointCloud>("local_cloud", 1000);
    pub_global_pts_ = n.advertise<sensor_msgs::PointCloud>("global_cloud", 1000);
}

void EskfEstimator::MainProcessThread() {
    std::cout<< "333 " << std::endl;
    while (!shutdown_) {
        bool is_update = false;
        sensor_msgs::ImuConstPtr p_imu = nullptr;
        sensor_msgs::PointCloudConstPtr p_lidar = nullptr;

        m_buf_.lock();
        // if (!imu_buf_.empty()) {
        //     p_imu = imu_buf_.front();
        //     double time = p_imu->header.stamp.toSec();
        //     Eigen::Vector3d acc(p_imu->linear_acceleration.x,
        //                         p_imu->linear_acceleration.y,
        //                         p_imu->linear_acceleration.z);
        //     Eigen::Vector3d gyro(p_imu->angular_velocity.x,
        //                          p_imu->angular_velocity.y,
        //                          p_imu->angular_velocity.z);
        //     ROS_INFO("deal with IMU. time stamp: %lf", time);
        //     PredictByImu(acc, gyro, time);
        //     is_update = true;
        //     imu_buf_.pop();
        // }

        if (!imu_buf_.empty() && !lidar_buf_.empty()) {
            p_lidar = lidar_buf_.front();
            double time_odom = p_lidar->header.stamp.toSec();
            if (imu_buf_.back()->header.stamp.toSec() >= time_odom) {
                while (1) {
                    p_imu = imu_buf_.front();
                    double time = p_imu->header.stamp.toSec();
                    Eigen::Vector3d acc(p_imu->linear_acceleration.x,
                                        p_imu->linear_acceleration.y,
                                        p_imu->linear_acceleration.z);
                    Eigen::Vector3d gyro(p_imu->angular_velocity.x,
                                         p_imu->angular_velocity.y,
                                         p_imu->angular_velocity.z);
                    ROS_INFO("deal with IMU. time stamp: %lf", time);
                    //處理所有小與當前雷達時間戳的imu數據
                    PredictByImu(acc, gyro, time);
                    imu_buf_.pop(); //彈出現在已經處理過的imu數據
                    if (imu_buf_.empty() || imu_buf_.front()->header.stamp.toSec() > time_odom) {
                        break;
                    }
                }

                double time = p_lidar->header.stamp.toSec();
                ROS_INFO("deal with lidar  time stamp: %lf", time);
                lidar_frame_points_.clear();
                //處理當前激光雷達幀
                for (int i = 0; i < p_lidar->points.size(); i++){
                    lidar_frame_points_.emplace_back(Eigen::Vector3d(p_lidar->points[i].x, p_lidar->points[i].y, p_lidar->points[i].z));
                }
                UpdateByLidar(time, lidar_frame_points_);
                is_update = true;
                lidar_buf_.pop();
            }
        }
        m_buf_.unlock();

        if(is_update) {
            PublishPose();
            PublishPoints();
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
}

void EskfEstimator::PredictByImu(const Eigen::Vector3d &imuAcc, const Eigen::Vector3d &imuGyro, const double &t) {

    if (!state_init_) {
        InitState(t);

    }
    // mean prediction
    double dt = t - state_.time;
    const Eigen::Vector3d acc =
            state_.q * (imuAcc - state_.ba) + gravity_;
    const Eigen::Vector3d omg = (imuGyro - state_.bg) * dt / 2.0; // 2.0
    Eigen::Quaterniond dq(1.0, omg(0), omg(1), omg(2)); // 2
    // state_.q = (state_.q * dq).normalized();
    // const Eigen::Vector3d acc1 =
    //         state_.q * (imuAcc - state_.ba) + gravity_;
    //
    // state_.p += state_.v * dt + 0.5 * acc * dt * dt; // 0.5
    // state_.v += acc * dt;
    // state_.time = t;

    // variance propogation
    Eigen::Matrix3d mI = Eigen::Matrix3d::Identity();
    Eigen::MatrixXd mF = Eigen::MatrixXd::Zero(StateIndex::STATE_TOTAL, StateIndex::STATE_TOTAL);

// dq
    mF.block<3, 3>(StateIndex::R, StateIndex::R) = mI - Utility::SkewSymmetric(imuGyro - state_.bg) * dt; // 3
    mF.block<3, 3>(StateIndex::R, StateIndex::BG) = -mI * dt; // 3
// dp

    mF.block<3, 3>(StateIndex::P, StateIndex::P) = mI; // 3
    mF.block<3, 3>(StateIndex::P, StateIndex::V) = dt * mI; // 3
// dv
    mF.block<3, 3>(StateIndex::V, StateIndex::R) =
            -dt * state_.q.toRotationMatrix() * Utility::SkewSymmetric(imuAcc - state_.ba); // 3
    mF.block<3, 3>(StateIndex::V, StateIndex::V) = mI; // 3
    mF.block<3, 3>(StateIndex::V, StateIndex::BA) = -state_.q.toRotationMatrix() * dt; // 3
// dba
    mF.block<3, 3>(StateIndex::BA, StateIndex::BA) = mI; // 3
// dbg
    mF.block<3, 3>(StateIndex::BG, StateIndex::BG) = mI; // 3

    Eigen::MatrixXd mU = Eigen::MatrixXd::Zero(StateIndex::STATE_TOTAL, StateNoiseIndex::NOISE_TOTAL);
// dq
    mU.block<3, 3>(StateIndex::R, StateNoiseIndex::ACC_NOISE) = -mI * dt; // 3
// dv
    mU.block<3, 3>(StateIndex::V, StateNoiseIndex::GYRO_NOISE) = -state_.q.toRotationMatrix() * dt; // 3
// dba
    mU.block<3, 3>(StateIndex::BA, StateNoiseIndex::ACC_RANDOM_WALK) = mI * dt; // 3
// dbg
    mU.block<3, 3>(StateIndex::BG, StateNoiseIndex::GYRO_RANDOM_WALK) = mI * dt; // 3

    state_.P = mF * state_.P * mF.transpose() + mU * Q_ * mU.transpose();
}

void EskfEstimator::InitMap(const std::vector<Eigen::Vector3d> &lidar_pts) {
    map_cloud_ = lidar_pts;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
    for(const auto& pt : map_cloud_) {
        pcl::PointXYZ pcl_pt;
        pcl_pt.x = pt(0);
        pcl_pt.y = pt(1);
        pcl_pt.z = pt(2);
        cloud->push_back(pcl_pt);
    }
    map_kdtree_.setInputCloud(cloud->makeShared());
    map_init_=true;



}

bool EskfEstimator::FindNearestPointInMap(const Eigen::Vector3d& local_pt, Eigen::Vector3d& map_pt, double threshold) {

    pcl::PointXYZ query_point;
    query_point.x = local_pt.x();
    query_point.y = local_pt.y();
    query_point.z = local_pt.z();

    std::vector<int> nearest_indices(1);
    std::vector<float> squared_distances(1);

    // Perform the nearest-neighbor search using the KDTree
    int num_neighbors = map_kdtree_.nearestKSearch(query_point, 1, nearest_indices, squared_distances);

    // Check if the distance is within the threshold
    if (num_neighbors > 0 && squared_distances[0] <= threshold) {
        const pcl::PointXYZ& nearest_point = map_kdtree_.getInputCloud()->points[nearest_indices[0]];

        // Return the nearest point as an Eigen vector
        map_pt = Eigen::Vector3d(nearest_point.x, nearest_point.y, nearest_point.z);
        ROS_INFO("map_pt.x: %f, map_pt.y: %f, map_pt.z: %f", map_pt.x(),map_pt.y(),map_pt.z());
        ROS_INFO("local_pt.x: %f, local_pt.y: %f, local_pt.z: %f", local_pt.x(),local_pt.y(),local_pt.z());

        return true;  // Found a valid nearest point
    }

    return false;  // No valid nearest point found
}

void EskfEstimator::IncreasePointToMap(const std::vector<Eigen::Vector3d> &unmatched_local_pts) {
    for (const auto& pt : unmatched_local_pts) {
        map_cloud_.push_back(pt);
    }

    // Now, update the KDTree with the new map cloud.
    pcl::PointCloud<pcl::PointXYZ>::Ptr map_cloud_pcl(new pcl::PointCloud<pcl::PointXYZ>());
    for (const auto& pt : map_cloud_) {

        pcl::PointXYZ pcl_pt(pt(0), pt(1), pt(2));
        map_cloud_pcl->points.push_back(pcl_pt);
    }
    // map_cloud_pcl->width = map_cloud_pcl->points.size();
    // map_cloud_pcl->height = 1;

    map_kdtree_.setInputCloud(map_cloud_pcl);

}
void EskfEstimator::UpdateByLidar(const double &time, const std::vector<Eigen::Vector3d> &lidar_pts) {

    // Your Code
    // 1. initialize the map (map_cloud_) ; build kdtree of map map_kdtree_
           // InitMap
    // Findthenearstinmap
    // 3. EKF update with every points

    // 4. maintain global map
           // IncreasePointToMap
    if (!map_init_) {
        InitMap(lidar_pts);
        return;

    }
    std::vector<Eigen::Vector3d> unmatched_local_pts;
    Eigen::Vector3d  map_pt;
    size_t count = 0;
    for (const auto& local_pt :lidar_pts) {
        std::cout<<"in xunhuan "<< std::endl;
        Eigen::Vector3d pt = state_.q * local_pt + state_.p;
        if (FindNearestPointInMap(pt, map_pt, 5.0)) {
            std::cout<< "FindNearestPointInMap" << std::endl;
            Eigen::Vector3d r = pt - map_pt;
            // Eigen::Vector3d r = local_pt -map_pt;
            Eigen::MatrixXd mH = Eigen::MatrixXd::Zero(3, StateIndex::STATE_TOTAL); // 9
            mH.block<3, 3>(0, StateIndex::R) = Utility::SkewSymmetric((state_.q.inverse() * state_.p)); // 3
            mH.block<3, 3>(0, StateIndex::P) = - state_.q.toRotationMatrix().inverse();
            Eigen::MatrixXd mS = mH * state_.P * mH.transpose() + Rm_;
            Eigen::MatrixXd mK = state_.P * mH.transpose() * mS.inverse();
            Eigen::VectorXd delta_state = mK * r;
            std::cout<<"delta_state: "<<delta_state<<std::endl;
            Eigen::Vector3d delta_R = delta_state.block<3, 1>(0,0);
            Eigen::Quaterniond dq(1, delta_R[0] /2, delta_R[1] /2 , delta_R[2] /2);
            state_.q = (state_.q * dq).normalized();
            state_.p += delta_state.block<3, 1>(3,0);
            state_.v += delta_state.block<3, 1>(6,0);
            state_.ba += delta_state.block<3, 1>(9,0);
            state_.bg += delta_state.block<3, 1>(12,0);
            state_.P = (Eigen::MatrixXd::Identity(StateIndex::STATE_TOTAL, StateIndex::STATE_TOTAL) - mK * mH) *
                       state_.P;
            

        } else {
            unmatched_local_pts.push_back(pt);
            count ++;
        }




    }
    std::cout<< "count: " << count <<std::endl;
    IncreasePointToMap(unmatched_local_pts);
    unmatched_local_pts.clear();



}

void EskfEstimator::UpdateByWheel(const double &time, const Eigen::Vector3d &wheelSpeed) {

    Eigen::VectorXd r(3); // 9
    r = wheelSpeed - state_.q.inverse() * state_.v; // 3

    Eigen::MatrixXd mH = Eigen::MatrixXd::Zero(3, StateIndex::STATE_TOTAL); // 9
    mH.block<3, 3>(0, StateIndex::R) = Utility::SkewSymmetric((state_.q.inverse() * state_.v)); // 3
    mH.block<3, 3>(0, StateIndex::V) = state_.q.toRotationMatrix().transpose(); // 3


    Eigen::MatrixXd mS = mH * state_.P * mH.transpose() + Rm_;
    Eigen::MatrixXd mK = state_.P * mH.transpose() * mS.inverse();
    Eigen::VectorXd dx = mK * r;

    state_.q = state_.q * Eigen::Quaterniond(1.0, dx(StateIndex::R) / 2.0, // 2.0
                                             dx(StateIndex::R + 1) / 2.0, dx(StateIndex::R + 2) / 2.0); // 2.0
    state_.q.normalize();
    state_.p += dx.segment<3>(StateIndex::P); // 3
    state_.v += dx.segment<3>(StateIndex::V); // 3
    state_.ba += dx.segment<3>(StateIndex::BA); // 3
    state_.bg += dx.segment<3>(StateIndex::BG); // 3
    state_.P = (Eigen::MatrixXd::Identity(StateIndex::STATE_TOTAL, StateIndex::STATE_TOTAL) - mK * mH) *
               state_.P;
}

void EskfEstimator::PublishPoints() {
    double time = state_.time;
    sensor_msgs::PointCloud global_cloud;
    global_cloud.header.frame_id = "world";
    global_cloud.header.stamp = ros::Time(time);
    for (auto &it : map_cloud_)
    {
        geometry_msgs::Point32 p;
        p.x = it(0);
        p.y = it(1);
        p.z = it(2);
        global_cloud.points.push_back(p);
    }
    pub_global_pts_.publish(global_cloud);

    sensor_msgs::PointCloud local_cloud;
    local_cloud.header.frame_id = "world";
    local_cloud.header.stamp = ros::Time(time);
    for (auto &it : lidar_frame_points_)
    {
        Eigen::Vector3d wp = state_.q * it + state_.p;
        geometry_msgs::Point32 p;
        p.x = wp(0);
        p.y = wp(1);
        p.z = wp(2);
        local_cloud.points.push_back(p);
    }
    pub_local_pts_.publish(local_cloud);
}

void EskfEstimator::PublishPose() {
    double time = state_.time;
    Eigen::Vector3d position = state_.p;
    Eigen::Quaterniond q = state_.q;
    Eigen::Vector3d velocity = state_.v;
    nav_msgs::Odometry odometry;
    odometry.header.frame_id = "world";
    odometry.header.stamp = ros::Time(time);
    odometry.pose.pose.position.x = position(0);
    odometry.pose.pose.position.y = position(1);
    odometry.pose.pose.position.z = position(2);
    odometry.pose.pose.orientation.x = q.x();
    odometry.pose.pose.orientation.y = q.y();
    odometry.pose.pose.orientation.z = q.z();
    odometry.pose.pose.orientation.w = q.w();
    odometry.twist.twist.linear.x = velocity(0);
    odometry.twist.twist.linear.y = velocity(1);
    odometry.twist.twist.linear.z = velocity(2);
    odometry.twist.covariance[0] = state_.ba.x();
    odometry.twist.covariance[1] = state_.ba.y();
    odometry.twist.covariance[2] = state_.ba.z();
    odometry.twist.covariance[3] = state_.bg.x();
    odometry.twist.covariance[4] = state_.bg.y();
    odometry.twist.covariance[5] = state_.bg.z();
    pub_odometry_.publish(odometry);

    ROS_INFO("IMU ACC Bias %lf %lf %lf GYRO Bias %lf %lf %lf \n", state_.ba.x(), state_.ba.y(), state_.ba.z(),
             state_.bg.x(), state_.bg.y(), state_.bg.z());

    geometry_msgs::PoseStamped pose_stamped;
    pose_stamped.header.frame_id = "world";
    pose_stamped.header.stamp = ros::Time(time);
    pose_stamped.pose = odometry.pose.pose;
    path_.poses.push_back(pose_stamped);
    pub_path_.publish(path_);

    static tf::TransformBroadcaster br;
    tf::Transform transform;
    tf::Quaternion q_tf;
    transform.setOrigin(tf::Vector3(state_.p(0),
                                    state_.p(1),
                                    state_.p(2)));
    q_tf.setW(state_.q.w());
    q_tf.setX(state_.q.x());
    q_tf.setY(state_.q.y());
    q_tf.setZ(state_.q.z());
    transform.setRotation(q_tf);
    br.sendTransform(tf::StampedTransform(transform, pose_stamped.header.stamp,
                                          "world", "eskf"));
}
