// SPDX-License-Identifier: BSD-2-Clause

#include <memory>
#include <iostream>

#include <rclcpp/rclcpp.hpp>
#include <pcl_ros/transforms.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/transform_broadcaster.h>

#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>

#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/approximate_voxel_grid.h>

#include <hdl_graph_slam/ros_utils.hpp>
#include <hdl_graph_slam/registrations.hpp>
#include <hdl_graph_slam/msg/scan_matching_status.hpp>

namespace hdl_graph_slam {

typedef struct {
  std::string points_topic;
  std::string odom_frame_id;
  std::string robot_odom_frame_id;
  std::string downsample_method;
  double keyframe_delta_trans;
  double keyframe_delta_angle;
  double keyframe_delta_time;
  double max_acceptable_trans;
  double max_acceptable_angle;
  double downsample_resolution;
  bool enable_imu_frontend;
  bool enable_robot_odometry_init_guess;
  bool transform_thresholding;
} ScanMatchingParams;

class ScanMatchingOdometryNode : public rclcpp::Node {
public:
  typedef pcl::PointXYZI PointT;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  ScanMatchingOdometryNode(const rclcpp::NodeOptions& options)
  : rclcpp::Node("scan_matching_odometry_node", options)
  {
    params.points_topic = declare_parameter<std::string>("points_topic", std::string("/velodyne_points"));
    params.odom_frame_id = declare_parameter<std::string>("odom_frame_id", std::string("odom"));
    params.robot_odom_frame_id = declare_parameter<std::string>("robot_odom_frame_id", std::string("odom"));
    params.downsample_method = declare_parameter<std::string>("downsample_method", std::string("VOXELGRID"));
    params.keyframe_delta_trans = declare_parameter<double>("keyframe_delta_trans", 0.25);
    params.keyframe_delta_angle = declare_parameter<double>("keyframe_delta_angle", 0.15);
    params.keyframe_delta_time = declare_parameter<double>("keyframe_delta_time", 1.0);
    params.max_acceptable_trans = declare_parameter<double>("max_acceptable_trans", 1.0);
    params.max_acceptable_angle = declare_parameter<double>("max_acceptable_angle", 1.0);
    params.downsample_resolution = declare_parameter<double>("downsample_resolution", 0.1);
    params.enable_imu_frontend = declare_parameter<bool>("enable_imu_frontend", false);
    params.enable_robot_odometry_init_guess = declare_parameter<bool>("enable_robot_odometry_init_guess", false);
    params.transform_thresholding = declare_parameter<bool>("transform_thresholding", false);

    // Registration parameters
    reg_params.registration_method = declare_parameter<std::string>("registration_method", std::string("NDT_OMP"));
    reg_params.reg_nn_search_method = declare_parameter<std::string>("reg_nn_search_method", std::string("DIRECT7"));
    reg_params.reg_transformation_epsilon = declare_parameter<double>("reg_transformation_epsilon", 0.01);
    reg_params.reg_max_correspondence_distance = declare_parameter<double>("reg_max_correspondence_distance", 2.5);
    reg_params.reg_resolution = declare_parameter<double>("reg_resolution", 1.0);
    reg_params.reg_num_threads = declare_parameter<int>("reg_num_threads", 0);
    reg_params.reg_maximum_iterations = declare_parameter<int>("reg_maximum_iterations", 64);
    reg_params.reg_correspondence_randomness = declare_parameter<int>("reg_correspondence_randomness", 20);
    reg_params.reg_max_optimizer_iterations = declare_parameter<int>("reg_max_optimizer_iterations", 20);
    reg_params.reg_use_reciprocal_correspondences = declare_parameter<bool>("reg_use_reciprocal_correspondences", false);

    initialize_params();

    tf_buffer = std::make_unique<tf2_ros::Buffer>(get_clock());
    tf_listener = std::make_shared<tf2_ros::TransformListener>(*tf_buffer);
    tf_broadcaster = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

    points_sub = create_subscription<sensor_msgs::msg::PointCloud2>("/filtered_points", 256, std::bind(&ScanMatchingOdometryNode::cloud_callback, this, std::placeholders::_1));
    if(params.enable_imu_frontend) {
      msf_pose_sub = create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>("/msf_core/pose", 1, std::bind(&ScanMatchingOdometryNode::msf_pose_callback, this, std::placeholders::_1));
      msf_pose_after_update_sub = create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>("/msf_core/pose_after_update", 1, std::bind(&ScanMatchingOdometryNode::msf_pose_after_update_callback, this, std::placeholders::_1));
    }

    read_until_pub = create_publisher<std_msgs::msg::Header>("/scan_matching_odometry/read_until", 32);
    odom_pub = create_publisher<nav_msgs::msg::Odometry>("/odom", 32);
    trans_pub = create_publisher<geometry_msgs::msg::TransformStamped>("/scan_matching_odometry/transform", 32);
    status_pub = create_publisher<msg::ScanMatchingStatus>("/scan_matching_odometry/status", 8);
    aligned_points_pub = create_publisher<sensor_msgs::msg::PointCloud2>("/aligned_points", 32);
  }

private:

  rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr param_cb;
  ScanMatchingParams params;
  RegistrationParams reg_params;

  /**
   * @brief initialize parameters
   */
  void initialize_params() {
    // select a downsample method (VOXELGRID, APPROX_VOXELGRID, NONE)
    if(params.downsample_method == "VOXELGRID") {
      std::cout << "downsample: VOXELGRID " << params.downsample_resolution << std::endl;
      auto voxelgrid = new pcl::VoxelGrid<PointT>();
      voxelgrid->setLeafSize(params.downsample_resolution, params.downsample_resolution, params.downsample_resolution);
      downsample_filter.reset(voxelgrid);
    } else if(params.downsample_method == "APPROX_VOXELGRID") {
      std::cout << "downsample: APPROX_VOXELGRID " << params.downsample_resolution << std::endl;
      pcl::ApproximateVoxelGrid<PointT>::Ptr approx_voxelgrid(new pcl::ApproximateVoxelGrid<PointT>());
      approx_voxelgrid->setLeafSize(params.downsample_resolution, params.downsample_resolution, params.downsample_resolution);
      downsample_filter = approx_voxelgrid;
    } else {
      if(params.downsample_method != "NONE") {
        std::cerr << "warning: unknown downsampling type (" << params.downsample_method << ")" << std::endl;
        std::cerr << "       : use passthrough filter" << std::endl;
      }
      std::cout << "downsample: NONE" << std::endl;
      pcl::PassThrough<PointT>::Ptr passthrough(new pcl::PassThrough<PointT>());
      downsample_filter = passthrough;
    }
    registration = select_registration_method(reg_params);
  }

  /**
   * @brief callback for point clouds
   * @param cloud_msg  point cloud msg
   */
  void cloud_callback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr cloud_msg) {
    if(!rclcpp::ok()) {
      return;
    }
    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>());
    pcl::fromROSMsg(*cloud_msg, *cloud);
    Eigen::Matrix4f pose = matching(cloud_msg->header.stamp, cloud);
    publish_odometry(cloud_msg->header.stamp, cloud_msg->header.frame_id, pose);

    // In offline estimation, point clouds until the published time will be supplied
    std_msgs::msg::Header read_until;
    read_until.frame_id = params.points_topic;
    read_until.stamp = rclcpp::Time(cloud_msg->header.stamp) + rclcpp::Duration(1, 0);
    read_until_pub->publish(read_until);

    read_until.frame_id = "/filtered_points";
    read_until_pub->publish(read_until);
  }

  void msf_pose_callback(const geometry_msgs::msg::PoseWithCovarianceStamped::ConstSharedPtr pose_msg) {
    msf_pose = pose_msg;
  }

  void msf_pose_after_update_callback(const geometry_msgs::msg::PoseWithCovarianceStamped::ConstSharedPtr pose_msg) {
    msf_pose_after_update = pose_msg;
  }

  /**
   * @brief downsample a point cloud
   * @param cloud  input cloud
   * @return downsampled point cloud
   */
  pcl::PointCloud<PointT>::ConstPtr downsample(const pcl::PointCloud<PointT>::ConstPtr& cloud) const {
    if(!downsample_filter) {
      return cloud;
    }

    pcl::PointCloud<PointT>::Ptr filtered(new pcl::PointCloud<PointT>());
    downsample_filter->setInputCloud(cloud);
    downsample_filter->filter(*filtered);

    return filtered;
  }

  /**
   * @brief estimate the relative pose between an input cloud and a keyframe cloud
   * @param stamp  the timestamp of the input cloud
   * @param cloud  the input cloud
   * @return the relative pose between the input cloud and the keyframe cloud
   */
  Eigen::Matrix4f matching(const rclcpp::Time& stamp, const pcl::PointCloud<PointT>::ConstPtr& cloud) {
    if(!keyframe) {
      prev_time = rclcpp::Time();
      prev_trans.setIdentity();
      keyframe_pose.setIdentity();
      keyframe_stamp = stamp;
      keyframe = downsample(cloud);
      registration->setInputTarget(keyframe);
      return Eigen::Matrix4f::Identity();
    }

    auto filtered = downsample(cloud);
    registration->setInputSource(filtered);

    std::string msf_source;
    Eigen::Isometry3f msf_delta = Eigen::Isometry3f::Identity();

    if(params.enable_imu_frontend) {
      if(msf_pose && rclcpp::Time(msf_pose->header.stamp) > keyframe_stamp && msf_pose_after_update && rclcpp::Time(msf_pose_after_update->header.stamp) > keyframe_stamp) {
        Eigen::Isometry3d pose0 = pose2isometry(msf_pose_after_update->pose.pose);
        Eigen::Isometry3d pose1 = pose2isometry(msf_pose->pose.pose);
        Eigen::Isometry3d delta = pose0.inverse() * pose1;

        msf_source = "imu";
        msf_delta = delta.cast<float>();
      } else {
        std::cerr << "msf data is too old" << std::endl;
      }
    } else if(params.enable_robot_odometry_init_guess && prev_time != rclcpp::Time(0)) {
      geometry_msgs::msg::TransformStamped transform;

      try {
        transform = tf_buffer->lookupTransform(cloud->header.frame_id, stamp, cloud->header.frame_id, prev_time, params.robot_odom_frame_id);
      } catch (tf2::LookupException ex) {
        RCLCPP_ERROR_STREAM(get_logger(), ex.what());
      } catch (tf2::ConnectivityException ex) {
        RCLCPP_ERROR_STREAM(get_logger(), ex.what());
      } catch (tf2::ExtrapolationException ex) {
        RCLCPP_ERROR_STREAM(get_logger(), ex.what());
      } catch (tf2::InvalidArgumentException ex) {
        RCLCPP_ERROR_STREAM(get_logger(), ex.what());
      }

      if(rclcpp::Time(transform.header.stamp) == rclcpp::Time(0)) {
        RCLCPP_WARN_STREAM(get_logger(), "failed to look up transform between " << cloud->header.frame_id << " and " << params.robot_odom_frame_id);
      } else {
        msf_source = "odometry";
        msf_delta = tf2isometry(transform.transform).cast<float>();
      }
    }

    pcl::PointCloud<PointT>::Ptr aligned(new pcl::PointCloud<PointT>());
    registration->align(*aligned, prev_trans * msf_delta.matrix());

    publish_scan_matching_status(stamp, cloud->header.frame_id, aligned, msf_source, msf_delta);

    if(!registration->hasConverged()) {
      return keyframe_pose * prev_trans;
    }

    Eigen::Matrix4f trans = registration->getFinalTransformation();
    Eigen::Matrix4f odom = keyframe_pose * trans;

    if(params.transform_thresholding) {
      Eigen::Matrix4f delta = prev_trans.inverse() * trans;
      double dx = delta.block<3, 1>(0, 3).norm();
      double da = std::acos(Eigen::Quaternionf(delta.block<3, 3>(0, 0)).w());

      if(dx > params.max_acceptable_trans || da > params.max_acceptable_angle) {
        return keyframe_pose * prev_trans;
      }
    }

    prev_time = stamp;
    prev_trans = trans;

    auto keyframe_trans = matrix2transform(stamp, keyframe_pose, params.odom_frame_id, "keyframe");
    tf_broadcaster->sendTransform(keyframe_trans);

    double delta_trans = trans.block<3, 1>(0, 3).norm();
    double delta_angle = std::acos(Eigen::Quaternionf(trans.block<3, 3>(0, 0)).w());
    double delta_time = (stamp - keyframe_stamp).seconds();
    if(delta_trans > params.keyframe_delta_trans || delta_angle > params.keyframe_delta_angle || delta_time > params.keyframe_delta_time) {
      keyframe = filtered;
      registration->setInputTarget(keyframe);

      keyframe_pose = odom;
      keyframe_stamp = stamp;
      prev_time = stamp;
      prev_trans.setIdentity();
    }

    if (aligned_points_pub->get_subscription_count() > 0) {
      pcl::transformPointCloud (*cloud, *aligned, odom);
      aligned->header.frame_id = params.odom_frame_id;
      sensor_msgs::msg::PointCloud2 output_msg;
      pcl::toROSMsg(*aligned, output_msg);
      aligned_points_pub->publish(output_msg);
    }

    return odom;
  }

  /**
   * @brief publish odometry
   * @param stamp  timestamp
   * @param pose   odometry pose to be published
   */
  void publish_odometry(const rclcpp::Time& stamp, const std::string& base_frame_id, const Eigen::Matrix4f& pose) {
    // publish transform stamped for IMU integration
    geometry_msgs::msg::TransformStamped odom_trans = matrix2transform(stamp, pose, params.odom_frame_id, base_frame_id);
    trans_pub->publish(odom_trans);

    // broadcast the transform over tf
    tf_broadcaster->sendTransform(odom_trans);

    // publish the transform
    nav_msgs::msg::Odometry odom;
    odom.header.stamp = stamp;
    odom.header.frame_id = params.odom_frame_id;

    odom.pose.pose.position.x = pose(0, 3);
    odom.pose.pose.position.y = pose(1, 3);
    odom.pose.pose.position.z = pose(2, 3);
    odom.pose.pose.orientation = odom_trans.transform.rotation;

    odom.child_frame_id = base_frame_id;
    odom.twist.twist.linear.x = 0.0;
    odom.twist.twist.linear.y = 0.0;
    odom.twist.twist.angular.z = 0.0;

    odom_pub->publish(odom);
  }

  /**
   * @brief publish scan matching status
   */
  void publish_scan_matching_status(const rclcpp::Time& stamp, const std::string& frame_id, pcl::PointCloud<pcl::PointXYZI>::ConstPtr aligned, const std::string& msf_source, const Eigen::Isometry3f& msf_delta) {
    if(!status_pub->get_subscription_count()) {
      return;
    }

    msg::ScanMatchingStatus status;
    status.header.frame_id = frame_id;
    status.header.stamp = stamp;
    status.has_converged = registration->hasConverged();
    status.matching_error = registration->getFitnessScore();

    const double max_correspondence_dist = 0.5;

    int num_inliers = 0;
    std::vector<int> k_indices;
    std::vector<float> k_sq_dists;
    for(int i=0; i<aligned->size(); i++) {
      const auto& pt = aligned->at(i);
      registration->getSearchMethodTarget()->nearestKSearch(pt, 1, k_indices, k_sq_dists);
      if(k_sq_dists[0] < max_correspondence_dist * max_correspondence_dist) {
        num_inliers++;
      }
    }
    status.inlier_fraction = static_cast<float>(num_inliers) / aligned->size();

    status.relative_pose = isometry2pose(Eigen::Isometry3f(registration->getFinalTransformation()).cast<double>());

    if(!msf_source.empty()) {
      status.prediction_labels.resize(1);
      status.prediction_labels[0].data = msf_source;

      status.prediction_errors.resize(1);
      Eigen::Isometry3f error = Eigen::Isometry3f(registration->getFinalTransformation()).inverse() * msf_delta;
      status.prediction_errors[0] = isometry2pose(error.cast<double>());
    }

    status_pub->publish(status);
  }

private:
  // ROS topics
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr points_sub;
  rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr msf_pose_sub;
  rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr msf_pose_after_update_sub;

  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub;
  rclcpp::Publisher<geometry_msgs::msg::TransformStamped>::SharedPtr trans_pub;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr aligned_points_pub;
  rclcpp::Publisher<hdl_graph_slam::msg::ScanMatchingStatus>::SharedPtr status_pub;
  rclcpp::Publisher<std_msgs::msg::Header>::SharedPtr read_until_pub;

  std::shared_ptr<tf2_ros::TransformListener> tf_listener;
  std::unique_ptr<tf2_ros::Buffer> tf_buffer;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster;

  // odometry calculation
  geometry_msgs::msg::PoseWithCovarianceStamped::ConstSharedPtr msf_pose;
  geometry_msgs::msg::PoseWithCovarianceStamped::ConstSharedPtr msf_pose_after_update;

  rclcpp::Time prev_time;
  Eigen::Matrix4f prev_trans;                  // previous estimated transform from keyframe
  Eigen::Matrix4f keyframe_pose;               // keyframe pose
  rclcpp::Time keyframe_stamp;                    // keyframe time
  pcl::PointCloud<PointT>::ConstPtr keyframe;  // keyframe point cloud

  //
  pcl::Filter<PointT>::Ptr downsample_filter;
  pcl::Registration<PointT, PointT>::Ptr registration;
};

}  // namespace hdl_graph_slam

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(hdl_graph_slam::ScanMatchingOdometryNode)
