// SPDX-License-Identifier: BSD-2-Clause

#include <memory>
#include <iostream>

#include <rclcpp/rclcpp.hpp>
#include <pcl_conversions/pcl_conversions.h>

#include <sensor_msgs/msg/point_cloud2.hpp>
#include <hdl_graph_slam/msg/floor_coeffs.hpp>

#include <pcl/common/transforms.h>
#include <pcl/features/normal_3d.h>
#include <pcl/search/impl/search.hpp>
#include <pcl/filters/impl/plane_clipper3D.hpp>
#include <pcl/filters/extract_indices.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>

namespace hdl_graph_slam {

typedef struct {
  std::string points_topic;
  double tilt_deg;              // approximate sensor tilt angle [deg]
  double sensor_height;         // approximate sensor height [m]
  double height_clip_range;     // points with heights in [sensor_height - height_clip_range, sensor_height + height_clip_range] will be used for floor detection
  double floor_normal_thresh;   // verticality check thresold for the detected floor plane [deg]
  double normal_filter_thresh;  // "non-"verticality check threshold [deg]
  int floor_pts_thresh;         // minimum number of support points of RANSAC to accept a detected floor plane
  bool use_normal_filtering;    // if true, points with "non-"vertical normals will be filtered before RANSAC
} FloorDetectionParams;

class FloorDetectionNode : public rclcpp::Node {
public:
  typedef pcl::PointXYZI PointT;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  FloorDetectionNode(const rclcpp::NodeOptions& options)
  : rclcpp::Node("floor_detection_node", options)
  {
    params.points_topic = declare_parameter<std::string>("points_topic", std::string("/filtered_points"));
    params.tilt_deg = declare_parameter<double>("tilt_deg", 0.0);
    params.sensor_height = declare_parameter<double>("sensor_height", 2.0);
    params.height_clip_range = declare_parameter<double>("height_clip_range", 1.0);
    params.floor_normal_thresh = declare_parameter<double>("floor_normal_thresh", 10.0);
    params.normal_filter_thresh = declare_parameter<double>("normal_filter_thresh", 20.0);
    params.floor_pts_thresh = declare_parameter<int>("floor_pts_thresh", 512);
    params.use_normal_filtering = declare_parameter<bool>("use_normal_filtering", true);

    points_sub = create_subscription<sensor_msgs::msg::PointCloud2>(params.points_topic, 256, std::bind(&FloorDetectionNode::cloud_callback, this, std::placeholders::_1));
    floor_pub = create_publisher<msg::FloorCoeffs>("/floor_detection/floor_coeffs", 32);
    read_until_pub = create_publisher<std_msgs::msg::Header>("/floor_detection/read_until", 32);
    floor_filtered_pub = create_publisher<sensor_msgs::msg::PointCloud2>("/floor_detection/floor_filtered_points", 32);
    floor_points_pub = create_publisher<sensor_msgs::msg::PointCloud2>("/floor_detection/floor_points", 32);
  }

private:
  rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr param_cb;
  FloorDetectionParams params;

  /**
   * @brief callback for point clouds
   * @param cloud_msg  point cloud msg
   */
  void cloud_callback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr cloud_msg) {
    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>());
    pcl::fromROSMsg(*cloud_msg, *cloud);

    if(cloud->empty()) {
      return;
    }

    // floor detection
    boost::optional<Eigen::Vector4f> floor = detect(cloud);

    // publish the detected floor coefficients
    msg::FloorCoeffs coeffs;
    coeffs.header = cloud_msg->header;
    if(floor) {
      coeffs.coeffs.resize(4);
      for(int i = 0; i < 4; i++) {
        coeffs.coeffs[i] = (*floor)[i];
      }
    }

    floor_pub->publish(coeffs);

    // for offline estimation
    std_msgs::msg::Header read_until;
    read_until.frame_id = params.points_topic;
    read_until.stamp = rclcpp::Time(cloud_msg->header.stamp) + rclcpp::Duration(1, 0);
    read_until_pub->publish(read_until);

    read_until.frame_id = "/filtered_points";
    read_until_pub->publish(read_until);
  }

  /**
   * @brief detect the floor plane from a point cloud
   * @param cloud  input cloud
   * @return detected floor plane coefficients
   */
  boost::optional<Eigen::Vector4f> detect(const pcl::PointCloud<PointT>::Ptr& cloud) const {
    // compensate the tilt rotation
    Eigen::Matrix4f tilt_matrix = Eigen::Matrix4f::Identity();
    tilt_matrix.topLeftCorner(3, 3) = Eigen::AngleAxisf(params.tilt_deg * M_PI / 180.0f, Eigen::Vector3f::UnitY()).toRotationMatrix();

    // filtering before RANSAC (height and normal filtering)
    pcl::PointCloud<PointT>::Ptr filtered(new pcl::PointCloud<PointT>);
    pcl::transformPointCloud(*cloud, *filtered, tilt_matrix);
    filtered = plane_clip(filtered, Eigen::Vector4f(0.0f, 0.0f, 1.0f, params.sensor_height + params.height_clip_range), false);
    filtered = plane_clip(filtered, Eigen::Vector4f(0.0f, 0.0f, 1.0f, params.sensor_height - params.height_clip_range), true);

    if(params.use_normal_filtering) {
      filtered = normal_filtering(filtered);
    }

    pcl::transformPointCloud(*filtered, *filtered, static_cast<Eigen::Matrix4f>(tilt_matrix.inverse()));

    if(floor_filtered_pub->get_subscription_count()) {
      filtered->header = cloud->header;
      sensor_msgs::msg::PointCloud2 output_msg;
      pcl::toROSMsg(*filtered, output_msg);
      floor_filtered_pub->publish(output_msg);
    }

    // too few points for RANSAC
    if(filtered->size() < params.floor_pts_thresh) {
      return boost::none;
    }

    // RANSAC
    pcl::SampleConsensusModelPlane<PointT>::Ptr model_p(new pcl::SampleConsensusModelPlane<PointT>(filtered));
    pcl::RandomSampleConsensus<PointT> ransac(model_p);
    ransac.setDistanceThreshold(0.1);
    ransac.computeModel();

    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    ransac.getInliers(inliers->indices);

    // too few inliers
    if(inliers->indices.size() < params.floor_pts_thresh) {
      return boost::none;
    }

    // verticality check of the detected floor's normal
    Eigen::Vector4f reference = tilt_matrix.inverse() * Eigen::Vector4f::UnitZ();

    Eigen::VectorXf coeffs;
    ransac.getModelCoefficients(coeffs);

    double dot = coeffs.head<3>().dot(reference.head<3>());
    if(std::abs(dot) < std::cos(params.floor_normal_thresh * M_PI / 180.0)) {
      // the normal is not vertical
      return boost::none;
    }

    // make the normal upward
    if(coeffs.head<3>().dot(Eigen::Vector3f::UnitZ()) < 0.0f) {
      coeffs *= -1.0f;
    }

    if(floor_points_pub->get_subscription_count()) {
      pcl::PointCloud<PointT>::Ptr inlier_cloud(new pcl::PointCloud<PointT>);
      pcl::ExtractIndices<PointT> extract;
      extract.setInputCloud(filtered);
      extract.setIndices(inliers);
      extract.filter(*inlier_cloud);
      inlier_cloud->header = cloud->header;
      sensor_msgs::msg::PointCloud2 output_msg;
      pcl::toROSMsg(*inlier_cloud, output_msg);
      floor_points_pub->publish(output_msg);
    }

    return Eigen::Vector4f(coeffs);
  }

  /**
   * @brief plane_clip
   * @param src_cloud
   * @param plane
   * @param negative
   * @return
   */
  pcl::PointCloud<PointT>::Ptr plane_clip(const pcl::PointCloud<PointT>::Ptr& src_cloud, const Eigen::Vector4f& plane, bool negative) const {
    pcl::PlaneClipper3D<PointT> clipper(plane);
    pcl::PointIndices::Ptr indices(new pcl::PointIndices);

    clipper.clipPointCloud3D(*src_cloud, indices->indices);

    pcl::PointCloud<PointT>::Ptr dst_cloud(new pcl::PointCloud<PointT>);

    pcl::ExtractIndices<PointT> extract;
    extract.setInputCloud(src_cloud);
    extract.setIndices(indices);
    extract.setNegative(negative);
    extract.filter(*dst_cloud);

    return dst_cloud;
  }

  /**
   * @brief filter points with non-vertical normals
   * @param cloud  input cloud
   * @return filtered cloud
   */
  pcl::PointCloud<PointT>::Ptr normal_filtering(const pcl::PointCloud<PointT>::Ptr& cloud) const {
    pcl::NormalEstimation<PointT, pcl::Normal> ne;
    ne.setInputCloud(cloud);

    pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);
    ne.setSearchMethod(tree);

    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    ne.setKSearch(10);
    ne.setViewPoint(0.0f, 0.0f, params.sensor_height);
    ne.compute(*normals);

    pcl::PointCloud<PointT>::Ptr filtered(new pcl::PointCloud<PointT>);
    filtered->reserve(cloud->size());

    for(int i = 0; i < cloud->size(); i++) {
      float dot = normals->at(i).getNormalVector3fMap().normalized().dot(Eigen::Vector3f::UnitZ());
      if(std::abs(dot) > std::cos(params.normal_filter_thresh * M_PI / 180.0)) {
        filtered->push_back(cloud->at(i));
      }
    }

    filtered->width = filtered->size();
    filtered->height = 1;
    filtered->is_dense = false;

    return filtered;
  }

private:
  // ROS topics
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr points_sub;
  rclcpp::Publisher<msg::FloorCoeffs>::SharedPtr floor_pub;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr floor_points_pub;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr floor_filtered_pub;
  rclcpp::Publisher<std_msgs::msg::Header>::SharedPtr read_until_pub;
};

}  // namespace hdl_graph_slam

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(hdl_graph_slam::FloorDetectionNode)
