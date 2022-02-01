// SPDX-License-Identifier: BSD-2-Clause

#include <string>

#include <rclcpp/rclcpp.hpp>
#include <pcl_ros/transforms.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <tf2_ros/transform_listener.h>

#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/approximate_voxel_grid.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/statistical_outlier_removal.h>

namespace hdl_graph_slam {

typedef struct {
  std::string downsample_method;
  std::string outlier_removal_method;
  std::string base_link_frame;
  double downsample_resolution;
  double statistical_stddev;
  double radius_radius;
  double distance_near_thresh;
  double distance_far_thresh;
  double scan_period;
  int statistical_mean_k;
  int radius_min_neighbors;
  bool deskewing;
} PrefilteringParams;

class PrefilteringNode : public rclcpp::Node {
public:
  typedef pcl::PointXYZI PointT;

  PrefilteringNode(const rclcpp::NodeOptions& options = rclcpp::NodeOptions())
  : rclcpp::Node("prefiltering_node", options)
  {
    params.downsample_method = declare_parameter<std::string>("downsample_method", std::string("VOXELGRID"));
    params.outlier_removal_method = declare_parameter<std::string>("outlier_removal_method", std::string("STATISTICAL"));
    params.base_link_frame = declare_parameter<std::string>("base_link_frame", std::string(""));
    params.downsample_resolution = declare_parameter<double>("downsample_resolution", 0.1);
    params.statistical_stddev = declare_parameter<double>("statistical_stddev", 1.0);
    params.radius_radius = declare_parameter<double>("radius_radius", 0.8);
    params.distance_near_thresh = declare_parameter<double>("distance_near_thresh", 1.0);
    params.distance_far_thresh = declare_parameter<double>("distance_far_thresh", 100.0);
    params.scan_period = declare_parameter<double>("scan_period", 0.1);
    params.statistical_mean_k = declare_parameter<int>("statistical_mean_k", 20);
    params.radius_min_neighbors = declare_parameter<int>("radius_min_neighbors", 2);
    params.deskewing = declare_parameter<bool>("deskewing", false);

    initialize_params();

    tf_buffer = std::make_unique<tf2_ros::Buffer>(get_clock());
    tf_listener = std::make_shared<tf2_ros::TransformListener>(*tf_buffer);

    points_sub = create_subscription<sensor_msgs::msg::PointCloud2>("/velodyne_points", 64, std::bind(&PrefilteringNode::cloud_callback, this, std::placeholders::_1));
    points_pub = create_publisher<sensor_msgs::msg::PointCloud2>("/filtered_points", 32);
    colored_pub = create_publisher<sensor_msgs::msg::PointCloud2>("/colored_points", 32);
    if(params.deskewing) {
      imu_sub = create_subscription<sensor_msgs::msg::Imu>("/imu/data", 1, std::bind(&PrefilteringNode::imu_callback, this, std::placeholders::_1));
    }
  }

private:
  rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr param_cb;

  void initialize_params() {

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
    }

    if(params.outlier_removal_method == "STATISTICAL") {
      std::cout << "outlier_removal: STATISTICAL " << params.statistical_mean_k << " - " << params.statistical_stddev << std::endl;

      pcl::StatisticalOutlierRemoval<PointT>::Ptr sor(new pcl::StatisticalOutlierRemoval<PointT>());
      sor->setMeanK(params.statistical_mean_k);
      sor->setStddevMulThresh(params.statistical_stddev);
      outlier_removal_filter = sor;
    } else if(params.outlier_removal_method == "RADIUS") {
      std::cout << "outlier_removal: RADIUS " << params.radius_radius << " - " << params.radius_min_neighbors << std::endl;

      pcl::RadiusOutlierRemoval<PointT>::Ptr rad(new pcl::RadiusOutlierRemoval<PointT>());
      rad->setRadiusSearch(params.radius_radius);
      rad->setMinNeighborsInRadius(params.radius_min_neighbors);
      outlier_removal_filter = rad;
    } else {
      std::cout << "outlier_removal: NONE" << std::endl;
    }
  }

  void imu_callback(const sensor_msgs::msg::Imu::ConstSharedPtr imu_msg) {
    imu_queue.push_back(imu_msg);
  }

  void cloud_callback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg) {
    pcl::PointCloud<PointT>::Ptr src_cloud(new pcl::PointCloud<PointT>);
    pcl::fromROSMsg(*msg, *src_cloud);
    if (src_cloud->empty()) {
      return;
    }

    src_cloud = deskewing(src_cloud);

    // if base_link_frame is defined, transform the input cloud to the frame
    if(!params.base_link_frame.empty()) {
      if(!tf_buffer->canTransform(params.base_link_frame, src_cloud->header.frame_id, rclcpp::Time(0))) {
        std::cerr << "failed to find transform between " << params.base_link_frame << " and " << src_cloud->header.frame_id << std::endl;
      }

      geometry_msgs::msg::TransformStamped transform;
      try {
        transform = tf_buffer->lookupTransform(params.base_link_frame, src_cloud->header.frame_id, rclcpp::Time(0));
      } catch (tf2::LookupException ex) {
        RCLCPP_ERROR_STREAM(get_logger(), ex.what());
      } catch (tf2::ConnectivityException ex) {
        RCLCPP_ERROR_STREAM(get_logger(), ex.what());
      } catch (tf2::ExtrapolationException ex) {
        RCLCPP_ERROR_STREAM(get_logger(), ex.what());
      } catch (tf2::InvalidArgumentException ex) {
        RCLCPP_ERROR_STREAM(get_logger(), ex.what());
      }

      pcl::PointCloud<PointT>::Ptr transformed(new pcl::PointCloud<PointT>());
      pcl_ros::transformPointCloud(*src_cloud, *transformed, transform);
      transformed->header.frame_id = params.base_link_frame;
      transformed->header.stamp = src_cloud->header.stamp;
      src_cloud = transformed;
    }

    pcl::PointCloud<PointT>::ConstPtr filtered = distance_filter(src_cloud);

    filtered = downsample(filtered);
    filtered = outlier_removal(filtered);
    sensor_msgs::msg::PointCloud2 output_msg;
    pcl::toROSMsg(*filtered, output_msg);
    points_pub->publish(output_msg);
  }

  pcl::PointCloud<PointT>::ConstPtr downsample(const pcl::PointCloud<PointT>::ConstPtr& cloud) const {
    if(!downsample_filter) {
      return cloud;
    }

    pcl::PointCloud<PointT>::Ptr filtered(new pcl::PointCloud<PointT>());
    downsample_filter->setInputCloud(cloud);
    downsample_filter->filter(*filtered);
    filtered->header = cloud->header;

    return filtered;
  }

  pcl::PointCloud<PointT>::ConstPtr outlier_removal(const pcl::PointCloud<PointT>::ConstPtr& cloud) const {
    if(!outlier_removal_filter) {
      return cloud;
    }

    pcl::PointCloud<PointT>::Ptr filtered(new pcl::PointCloud<PointT>());
    outlier_removal_filter->setInputCloud(cloud);
    outlier_removal_filter->filter(*filtered);
    filtered->header = cloud->header;

    return filtered;
  }

  pcl::PointCloud<PointT>::ConstPtr distance_filter(const pcl::PointCloud<PointT>::ConstPtr& cloud) const {
    pcl::PointCloud<PointT>::Ptr filtered(new pcl::PointCloud<PointT>());
    filtered->reserve(cloud->size());

    std::copy_if(cloud->begin(), cloud->end(), std::back_inserter(filtered->points), [&](const PointT& p) {
      double d = p.getVector3fMap().norm();
      return d > params.distance_near_thresh && d < params.distance_far_thresh;
    });

    filtered->width = filtered->size();
    filtered->height = 1;
    filtered->is_dense = false;

    filtered->header = cloud->header;

    return filtered;
  }

  pcl::PointCloud<PointT>::Ptr deskewing(const pcl::PointCloud<PointT>::Ptr& cloud) {
    auto stamp = pcl_conversions::fromPCL(cloud->header.stamp);
    if(imu_queue.empty()) {
      return cloud;
    }

    // the color encodes the point number in the point sequence
    if(colored_pub->get_subscription_count()) {
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored(new pcl::PointCloud<pcl::PointXYZRGB>());
      colored->header = cloud->header;
      colored->is_dense = cloud->is_dense;
      colored->width = cloud->width;
      colored->height = cloud->height;
      colored->resize(cloud->size());

      for(int i = 0; i < cloud->size(); i++) {
        double t = static_cast<double>(i) / cloud->size();
        colored->at(i).getVector4fMap() = cloud->at(i).getVector4fMap();
        colored->at(i).r = 255 * t;
        colored->at(i).g = 128;
        colored->at(i).b = 255 * (1 - t);
      }

      sensor_msgs::msg::PointCloud2 output_msg;
      pcl::toROSMsg(*colored, output_msg);
      colored_pub->publish(output_msg);
    }

    auto imu_msg = imu_queue.front();

    auto loc = imu_queue.begin();
    for(; loc != imu_queue.end(); loc++) {
      imu_msg = (*loc);
      auto ros_msg_stamp = rclcpp::Time((*loc)->header.stamp);
      if(ros_msg_stamp > stamp) {
        break;
      }
    }

    imu_queue.erase(imu_queue.begin(), loc);

    Eigen::Vector3f ang_v(imu_msg->angular_velocity.x, imu_msg->angular_velocity.y, imu_msg->angular_velocity.z);
    ang_v *= -1;

    pcl::PointCloud<PointT>::Ptr deskewed(new pcl::PointCloud<PointT>());
    deskewed->header = cloud->header;
    deskewed->is_dense = cloud->is_dense;
    deskewed->width = cloud->width;
    deskewed->height = cloud->height;
    deskewed->resize(cloud->size());

    for(int i = 0; i < cloud->size(); i++) {
      const auto& pt = cloud->at(i);

      // TODO: transform IMU data into the LIDAR frame
      double delta_t = params.scan_period * static_cast<double>(i) / cloud->size();
      Eigen::Quaternionf delta_q(1, delta_t / 2.0 * ang_v[0], delta_t / 2.0 * ang_v[1], delta_t / 2.0 * ang_v[2]);
      Eigen::Vector3f pt_ = delta_q.inverse() * pt.getVector3fMap();

      deskewed->at(i) = cloud->at(i);
      deskewed->at(i).getVector3fMap() = pt_;
    }

    return deskewed;
  }

private:
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub;
  std::vector<sensor_msgs::msg::Imu::ConstSharedPtr> imu_queue;

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr points_sub;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr points_pub;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr colored_pub;

  std::shared_ptr<tf2_ros::TransformListener> tf_listener;
  std::unique_ptr<tf2_ros::Buffer> tf_buffer;

  pcl::Filter<PointT>::Ptr downsample_filter;
  pcl::Filter<PointT>::Ptr outlier_removal_filter;
  PrefilteringParams params;
};

}  // namespace hdl_graph_slam

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(hdl_graph_slam::PrefilteringNode)
