// SPDX-License-Identifier: BSD-2-Clause

#include <ctime>
#include <mutex>
#include <atomic>
#include <memory>
#include <iomanip>
#include <iostream>
#include <unordered_map>
#include <boost/format.hpp>
#include <boost/thread.hpp>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>
#include <Eigen/Dense>
#include <pcl/io/pcd_io.h>

#include <rclcpp/rclcpp.hpp>
#include <geodesy/utm.h>
#include <geodesy/wgs84.h>
#include <pcl_conversions/pcl_conversions.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>

#include <nav_msgs/msg/odometry.hpp>
#include <nmea_msgs/msg/sentence.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/nav_sat_fix.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geographic_msgs/msg/geo_point_stamped.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <hdl_graph_slam/msg/floor_coeffs.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/vector3_stamped.hpp>
#include <geometry_msgs/msg/quaternion_stamped.hpp>

#include <hdl_graph_slam/srv/save_map.hpp>
#include <hdl_graph_slam/srv/dump_graph.hpp>

#include <hdl_graph_slam/ros_utils.hpp>
#include <hdl_graph_slam/ros_time_hash.hpp>

#include <hdl_graph_slam/graph_slam.hpp>
#include <hdl_graph_slam/keyframe.hpp>
#include <hdl_graph_slam/keyframe_updater.hpp>
#include <hdl_graph_slam/loop_detector.hpp>
#include <hdl_graph_slam/information_matrix_calculator.hpp>
#include <hdl_graph_slam/map_cloud_generator.hpp>
#include <hdl_graph_slam/nmea_sentence_parser.hpp>

#include <g2o/types/slam3d/edge_se3.h>
#include <g2o/types/slam3d/vertex_se3.h>
#include <g2o/edge_se3_plane.hpp>
#include <g2o/edge_se3_priorxy.hpp>
#include <g2o/edge_se3_priorxyz.hpp>
#include <g2o/edge_se3_priorvec.hpp>
#include <g2o/edge_se3_priorquat.hpp>

namespace hdl_graph_slam {

typedef struct {
  std::string map_frame_id;
  std::string odom_frame_id;
  std::string points_topic;
  std::string g2o_solver_type;
  std::string odometry_edge_robust_kernel;
  std::string gps_edge_robust_kernel;
  std::string imu_orientation_edge_robust_kernel;
  std::string imu_acceleration_edge_robust_kernel;
  std::string loop_closure_edge_robust_kernel;
  std::string floor_edge_robust_kernel;
  std::string fix_first_node_stddev;
  double map_cloud_resolution;
  double gps_time_offset;
  double gps_edge_stddev_xy;
  double gps_edge_stddev_z;
  double floor_edge_stddev;
  double imu_time_offset;
  double imu_orientation_edge_stddev;
  double imu_acceleration_edge_stddev;
  double graph_update_interval;
  double map_cloud_update_interval;
  double odometry_edge_robust_kernel_size;
  double gps_edge_robust_kernel_size;
  double imu_orientation_edge_robust_kernel_size;
  double imu_acceleration_edge_robust_kernel_size;
  double loop_closure_edge_robust_kernel_size;
  double floor_edge_robust_kernel_size;
  int max_keyframes_per_update;
  int g2o_solver_num_iterations;
  bool enable_imu_orientation;
  bool enable_imu_acceleration;
  bool enable_gps;
  bool fix_first_node;
  bool fix_first_node_adaptive;
} HdlGraphSlamParams;

class HdlGraphSlamNode : public rclcpp::Node {
public:
  typedef pcl::PointXYZI PointT;
  typedef message_filters::sync_policies::ApproximateTime<nav_msgs::msg::Odometry, sensor_msgs::msg::PointCloud2> ApproxSyncPolicy;

  HdlGraphSlamNode(const rclcpp::NodeOptions& options)
  : rclcpp::Node("hdl_graph_slam_node", options)
  {
    trans_odom2map.setIdentity();

    // Main parameters
    params.map_frame_id                             = declare_parameter<std::string>("map_frame_id", std::string("map"));
    params.odom_frame_id                            = declare_parameter<std::string>("odom_frame_id", std::string("odom"));
    params.g2o_solver_type                          = declare_parameter<std::string>("g2o_solver_type", std::string("lm_var"));
    params.odometry_edge_robust_kernel              = declare_parameter<std::string>("odometry_edge_robust_kernel", std::string("NONE"));
    params.gps_edge_robust_kernel                   = declare_parameter<std::string>("gps_edge_robust_kernel", std::string("NONE"));
    params.imu_orientation_edge_robust_kernel       = declare_parameter<std::string>("imu_orientation_edge_robust_kernel", std::string("NONE"));
    params.imu_acceleration_edge_robust_kernel      = declare_parameter<std::string>("imu_acceleration_edge_robust_kernel", std::string("NONE"));
    params.floor_edge_robust_kernel                 = declare_parameter<std::string>("floor_edge_robust_kernel", std::string("NONE"));
    params.loop_closure_edge_robust_kernel          = declare_parameter<std::string>("loop_closure_edge_robust_kernel", std::string("NONE"));
    params.fix_first_node_stddev                    = declare_parameter<std::string>("fix_first_node_stddev", std::string("1 1 1 1 1 1"));
    params.points_topic                             = declare_parameter<std::string>("points_topic", std::string("/velodyne_points"));
    params.map_cloud_resolution                     = declare_parameter<double>("map_cloud_resolution", 0.05);
    params.graph_update_interval                    = declare_parameter<double>("graph_update_interval", 3.0);
    params.map_cloud_update_interval                = declare_parameter<double>("map_cloud_update_interval", 10.0);
    params.odometry_edge_robust_kernel_size         = declare_parameter<double>("odometry_edge_robust_kernel_size", 1.0);
    params.gps_edge_robust_kernel_size              = declare_parameter<double>("gps_edge_robust_kernel_size", 1.0);
    params.imu_orientation_edge_robust_kernel_size  = declare_parameter<double>("imu_orientation_edge_robust_kernel_size", 1.0);
    params.imu_acceleration_edge_robust_kernel_size = declare_parameter<double>("imu_acceleration_edge_robust_kernel_size", 1.0);
    params.loop_closure_edge_robust_kernel_size     = declare_parameter<double>("loop_closure_edge_robust_kernel_size", 1.0);
    params.floor_edge_robust_kernel_size            = declare_parameter<double>("floor_edge_robust_kernel_size", 1.0);
    params.gps_time_offset                          = declare_parameter<double>("gps_time_offset", 0.0);
    params.gps_edge_stddev_xy                       = declare_parameter<double>("gps_edge_stddev_xy", 10000.0);
    params.gps_edge_stddev_z                        = declare_parameter<double>("gps_edge_stddev_z", 10.0);
    params.floor_edge_stddev                        = declare_parameter<double>("floor_edge_stddev", 10.0);
    params.imu_time_offset                          = declare_parameter<double>("imu_time_offset", 0.0);
    params.imu_orientation_edge_stddev              = declare_parameter<double>("imu_orientation_edge_stddev", 0.1);
    params.imu_acceleration_edge_stddev             = declare_parameter<double>("imu_acceleration_edge_stddev", 3.0);
    params.max_keyframes_per_update                 = declare_parameter<int>("max_keyframes_per_update", 10);
    params.g2o_solver_num_iterations                = declare_parameter<int>("g2o_solver_num_iterations", 1024);
    params.enable_gps                               = declare_parameter<bool>("enable_gps", true);
    params.fix_first_node                           = declare_parameter<bool>("fix_first_node", false);
    params.fix_first_node_adaptive                  = declare_parameter<bool>("fix_first_node_adaptive", true);
    params.enable_imu_orientation                   = declare_parameter<bool>("enable_imu_orientation", false);
    params.enable_imu_acceleration                  = declare_parameter<bool>("enable_imu_acceleration", false);

    map_publish_timer = rclcpp::create_timer(this, get_clock(), std::chrono::milliseconds((int)(params.map_cloud_update_interval * 1000)), std::bind(&HdlGraphSlamNode::map_points_publish_timer_callback, this));
    optimization_timer = rclcpp::create_timer(this, get_clock(), std::chrono::milliseconds((int)(params.graph_update_interval * 1000)), std::bind(&HdlGraphSlamNode::optimization_timer_callback, this));
    graph_slam.reset(new GraphSLAM(params.g2o_solver_type));
    if(params.enable_gps) {
      gps_sub = create_subscription<geographic_msgs::msg::GeoPointStamped>("/gps/geopoint", 1024, std::bind(&HdlGraphSlamNode::gps_callback, this, std::placeholders::_1));
      nmea_sub = create_subscription<nmea_msgs::msg::Sentence>("/gpsimu_driver/nmea_sentence", 1024, std::bind(&HdlGraphSlamNode::nmea_callback, this, std::placeholders::_1));
      navsat_sub = create_subscription<sensor_msgs::msg::NavSatFix>("/gps/navsat", 1024, std::bind(&HdlGraphSlamNode::navsat_callback, this, std::placeholders::_1));
    }

    // Keyframe updater params
    KeyframeUpdaterParams keyframe_params;
    keyframe_params.keyframe_delta_trans = declare_parameter<double>("keyframe_delta_trans", 2.0);
    keyframe_params.keyframe_delta_angle = declare_parameter<double>("keyframe_delta_angle", 2.0);

    // Loop detector params
    LoopDetectorParams loop_params;
    loop_params.distance_thresh = declare_parameter<double>("distance_thresh", 5.0);
    loop_params.accum_distance_thresh = declare_parameter<double>("accum_distance_thresh", 8.0);
    loop_params.min_edge_interval = declare_parameter<double>("min_edge_interval", 5.0);
    loop_params.fitness_score_max_range = declare_parameter<double>("fitness_score_max_range", std::numeric_limits<double>::max());
    loop_params.fitness_score_thresh = declare_parameter<double>("fitness_score_thresh", 0.5);

    // Registration params
    RegistrationParams reg_params;
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

    // Information matrix params
    InformationMatrixParams inf_mat_params;
    inf_mat_params.const_stddev_x = declare_parameter<double>("const_stddev_x", 0.5);
    inf_mat_params.const_stddev_q = declare_parameter<double>("const_stddev_q", 0.1);
    inf_mat_params.var_gain_a = declare_parameter<double>("var_gain_a", 20.0);
    inf_mat_params.min_stddev_x = declare_parameter<double>("min_stddev_x", 0.1);
    inf_mat_params.max_stddev_x = declare_parameter<double>("max_stddev_x", 5.0);
    inf_mat_params.min_stddev_q = declare_parameter<double>("min_stddev_q", 0.05);
    inf_mat_params.max_stddev_q = declare_parameter<double>("max_stddev_q", 0.2);
    inf_mat_params.use_const_inf_matrix = declare_parameter<bool>("use_const_inf_matrix", false);

    tf_buffer = std::make_unique<tf2_ros::Buffer>(get_clock());
    tf_listener = std::make_shared<tf2_ros::TransformListener>(*tf_buffer);

    anchor_node = nullptr;
    anchor_edge = nullptr;
    floor_plane_node = nullptr;

    // subscribers
    rmw_qos_profile_t qos =
    {
      RMW_QOS_POLICY_HISTORY_KEEP_LAST,
      1000,
      RMW_QOS_POLICY_RELIABILITY_RELIABLE,
      RMW_QOS_POLICY_DURABILITY_VOLATILE,
      RMW_QOS_DEADLINE_DEFAULT,
      RMW_QOS_LIFESPAN_DEFAULT,
      RMW_QOS_POLICY_LIVELINESS_SYSTEM_DEFAULT,
      RMW_QOS_LIVELINESS_LEASE_DURATION_DEFAULT,
      false
    };
    odom_sub.reset(new message_filters::Subscriber<nav_msgs::msg::Odometry>(this, "/odom", qos));
    cloud_sub.reset(new message_filters::Subscriber<sensor_msgs::msg::PointCloud2>(rclcpp::Node::SharedPtr(this), "/filtered_points", qos));
    sync.reset(new message_filters::Synchronizer<ApproxSyncPolicy>(ApproxSyncPolicy(32), *odom_sub, *cloud_sub));
    sync->registerCallback(std::bind(&HdlGraphSlamNode::cloud_callback, this, std::placeholders::_1, std::placeholders::_2));
    imu_sub = create_subscription<sensor_msgs::msg::Imu>("/gpsimu_driver/imu_data", 1024, std::bind(&HdlGraphSlamNode::imu_callback, this, std::placeholders::_1));
    floor_sub = create_subscription<msg::FloorCoeffs>("/floor_detection/floor_coeffs", 1024, std::bind(&HdlGraphSlamNode::floor_coeffs_callback, this, std::placeholders::_1));

    // publishers
    markers_pub = create_publisher<visualization_msgs::msg::MarkerArray>("/hdl_graph_slam/markers", 16);
    odom2map_pub = create_publisher<geometry_msgs::msg::TransformStamped>("/hdl_graph_slam/odom2pub", 16);
    map_points_pub = create_publisher<sensor_msgs::msg::PointCloud2>("/hdl_graph_slam/map_points", 1);
    read_until_pub = create_publisher<std_msgs::msg::Header>("/hdl_graph_slam/read_until", 32);

    // service servers
    dump_service_server = create_service<srv::DumpGraph>("/hdl_graph_slam/dump", std::bind(&HdlGraphSlamNode::dump_service, this, std::placeholders::_1, std::placeholders::_2));
    save_map_service_server = create_service<srv::SaveMap>("/hdl_graph_slam/save_map", std::bind(&HdlGraphSlamNode::save_map_service, this, std::placeholders::_1, std::placeholders::_2));

    graph_updated = false;
    nmea_parser.reset(new NmeaSentenceParser());
    map_cloud_generator.reset(new MapCloudGenerator());
    keyframe_updater.reset(new KeyframeUpdater(keyframe_params));
    loop_detector.reset(new LoopDetector(loop_params, reg_params));
    inf_calclator.reset(new InformationMatrixCalculator(inf_mat_params));
  }

private:

  rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr param_cb;
  HdlGraphSlamParams params;

  /**
   * @brief received point clouds are pushed to #keyframe_queue
   * @param odom_msg
   * @param cloud_msg
   */
  void cloud_callback(const nav_msgs::msg::Odometry::ConstSharedPtr odom_msg, const sensor_msgs::msg::PointCloud2::ConstSharedPtr cloud_msg) {
    const rclcpp::Time& stamp = cloud_msg->header.stamp;
    Eigen::Isometry3d odom = odom2isometry(odom_msg);

    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>());
    pcl::fromROSMsg(*cloud_msg, *cloud);
    if(base_frame_id.empty()) {
      base_frame_id = cloud_msg->header.frame_id;
    }

    if(!keyframe_updater->update(odom)) {
      std::lock_guard<std::mutex> lock(keyframe_queue_mutex);
      if(keyframe_queue.empty()) {
        std_msgs::msg::Header read_until;
        read_until.stamp = stamp + rclcpp::Duration(10, 0);
        read_until.frame_id = points_topic;
        read_until_pub->publish(read_until);
        read_until.frame_id = "/filtered_points";
        read_until_pub->publish(read_until);
      }

      return;
    }

    double accum_d = keyframe_updater->get_accum_distance();
    KeyFrame::Ptr keyframe(new KeyFrame(stamp, odom, accum_d, cloud));

    std::lock_guard<std::mutex> lock(keyframe_queue_mutex);
    keyframe_queue.push_back(keyframe);
  }

  /**
   * @brief this method adds all the keyframes in #keyframe_queue to the pose graph (odometry edges)
   * @return if true, at least one keyframe was added to the pose graph
   */
  bool flush_keyframe_queue() {
    std::lock_guard<std::mutex> lock(keyframe_queue_mutex);
    if(keyframe_queue.empty()) {
      return false;
    }

    trans_odom2map_mutex.lock();
    Eigen::Isometry3d odom2map(trans_odom2map.cast<double>());
    trans_odom2map_mutex.unlock();

    int num_processed = 0;
    for(int i = 0; i < std::min<int>(keyframe_queue.size(), params.max_keyframes_per_update); i++) {
      num_processed = i;

      const auto& keyframe = keyframe_queue[i];
      // new_keyframes will be tested later for loop closure
      new_keyframes.push_back(keyframe);

      // add pose node
      Eigen::Isometry3d odom = odom2map * keyframe->odom;
      keyframe->node = graph_slam->add_se3_node(odom);
      keyframe_hash[keyframe->stamp] = keyframe;

      // fix the first node
      if(keyframes.empty() && new_keyframes.size() == 1) {
        if(params.fix_first_node) {
          Eigen::MatrixXd inf = Eigen::MatrixXd::Identity(6, 6);
          std::stringstream sst(params.fix_first_node_stddev);
          for(int i = 0; i < 6; i++) {
            double stddev = 1.0;
            sst >> stddev;
            inf(i, i) = 1.0 / stddev;
          }

          anchor_node = graph_slam->add_se3_node(Eigen::Isometry3d::Identity());
          anchor_node->setFixed(true);
          anchor_edge = graph_slam->add_se3_edge(anchor_node, keyframe->node, Eigen::Isometry3d::Identity(), inf);
        }
      }

      if(i == 0 && keyframes.empty()) {
        continue;
      }

      // add edge between consecutive keyframes
      const auto& prev_keyframe = i == 0 ? keyframes.back() : keyframe_queue[i - 1];

      Eigen::Isometry3d relative_pose = keyframe->odom.inverse() * prev_keyframe->odom;
      Eigen::MatrixXd information = inf_calclator->calc_information_matrix(keyframe->cloud, prev_keyframe->cloud, relative_pose);
      auto edge = graph_slam->add_se3_edge(keyframe->node, prev_keyframe->node, relative_pose, information);
      graph_slam->add_robust_kernel(edge, params.odometry_edge_robust_kernel, params.odometry_edge_robust_kernel_size);
    }

    std_msgs::msg::Header read_until;
    read_until.stamp = keyframe_queue[num_processed]->stamp + rclcpp::Duration(10, 0);
    read_until.frame_id = points_topic;
    read_until_pub->publish(read_until);
    read_until.frame_id = "/filtered_points";
    read_until_pub->publish(read_until);

    keyframe_queue.erase(keyframe_queue.begin(), keyframe_queue.begin() + num_processed + 1);
    return true;
  }

  void nmea_callback(const nmea_msgs::msg::Sentence::ConstSharedPtr nmea_msg) {
    GPRMC grmc = nmea_parser->parse(nmea_msg->sentence);

    if(grmc.status != 'A') {
      return;
    }

    geographic_msgs::msg::GeoPointStamped::SharedPtr gps_msg(new geographic_msgs::msg::GeoPointStamped());
    gps_msg->header = nmea_msg->header;
    gps_msg->position.latitude = grmc.latitude;
    gps_msg->position.longitude = grmc.longitude;
    gps_msg->position.altitude = NAN;

    gps_callback(gps_msg);
  }

  void navsat_callback(const sensor_msgs::msg::NavSatFix::ConstSharedPtr navsat_msg) {
    geographic_msgs::msg::GeoPointStamped::SharedPtr gps_msg(new geographic_msgs::msg::GeoPointStamped());
    gps_msg->header = navsat_msg->header;
    gps_msg->position.latitude = navsat_msg->latitude;
    gps_msg->position.longitude = navsat_msg->longitude;
    gps_msg->position.altitude = navsat_msg->altitude;
    gps_callback(gps_msg);
  }

  /**
   * @brief received gps data is added to #gps_queue
   * @param gps_msg
   */
  void gps_callback(const geographic_msgs::msg::GeoPointStamped::SharedPtr gps_msg) {
    std::lock_guard<std::mutex> lock(gps_queue_mutex);
    rclcpp::Duration offset(std::chrono::milliseconds((int)(1000.0 * params.gps_time_offset)));
    gps_msg->header.stamp = rclcpp::Time(gps_msg->header.stamp) + offset;
    gps_queue.push_back(gps_msg);
  }

  /**
   * @brief
   * @return
   */
  bool flush_gps_queue() {
    std::lock_guard<std::mutex> lock(gps_queue_mutex);

    if(keyframes.empty() || gps_queue.empty()) {
      return false;
    }

    bool updated = false;
    auto gps_cursor = gps_queue.begin();

    for(auto& keyframe : keyframes) {
      if(keyframe->stamp > gps_queue.back()->header.stamp) {
        break;
      }

      if(keyframe->stamp < (*gps_cursor)->header.stamp || keyframe->utm_coord) {
        continue;
      }

      // find the gps data which is closest to the keyframe
      auto closest_gps = gps_cursor;
      for(auto gps = gps_cursor; gps != gps_queue.end(); gps++) {
        auto dt = (rclcpp::Time((*closest_gps)->header.stamp) - keyframe->stamp).seconds();
        auto dt2 = (rclcpp::Time((*gps)->header.stamp) - keyframe->stamp).seconds();
        if(std::abs(dt) < std::abs(dt2)) {
          break;
        }

        closest_gps = gps;
      }

      // if the time residual between the gps and keyframe is too large, skip it
      gps_cursor = closest_gps;
      if(0.2 < std::abs((rclcpp::Time((*closest_gps)->header.stamp) - keyframe->stamp).seconds())) {
        continue;
      }

      // convert (latitude, longitude, altitude) -> (easting, northing, altitude) in UTM coordinate
      geodesy::UTMPoint utm;
      geodesy::fromMsg((*closest_gps)->position, utm);
      Eigen::Vector3d xyz(utm.easting, utm.northing, utm.altitude);

      // the first gps data position will be the origin of the map
      if(!zero_utm) {
        zero_utm = xyz;
      }
      xyz -= (*zero_utm);

      keyframe->utm_coord = xyz;

      g2o::OptimizableGraph::Edge* edge;
      if(std::isnan(xyz.z())) {
        Eigen::Matrix2d information_matrix = Eigen::Matrix2d::Identity() / params.gps_edge_stddev_xy;
        edge = graph_slam->add_se3_prior_xy_edge(keyframe->node, xyz.head<2>(), information_matrix);
      } else {
        Eigen::Matrix3d information_matrix = Eigen::Matrix3d::Identity();
        information_matrix.block<2, 2>(0, 0) /= params.gps_edge_stddev_xy;
        information_matrix(2, 2) /= params.gps_edge_stddev_z;
        edge = graph_slam->add_se3_prior_xyz_edge(keyframe->node, xyz, information_matrix);
      }
      graph_slam->add_robust_kernel(edge, params.gps_edge_robust_kernel, params.gps_edge_robust_kernel_size);

      updated = true;
    }

    auto remove_loc = std::upper_bound(gps_queue.begin(), gps_queue.end(), keyframes.back()->stamp, [=](const rclcpp::Time& stamp, const geographic_msgs::msg::GeoPointStamped::ConstSharedPtr geopoint) { return stamp < rclcpp::Time(geopoint->header.stamp); });
    gps_queue.erase(gps_queue.begin(), remove_loc);
    return updated;
  }

  void imu_callback(const sensor_msgs::msg::Imu::SharedPtr imu_msg) {
    if(!params.enable_imu_orientation && !params.enable_imu_acceleration) {
      return;
    }

    std::lock_guard<std::mutex> lock(imu_queue_mutex);
    rclcpp::Duration offset(std::chrono::milliseconds((int)(1000.0 * params.imu_time_offset)));
    imu_msg->header.stamp = rclcpp::Time(imu_msg->header.stamp) + offset;
    imu_queue.push_back(imu_msg);
  }

  bool flush_imu_queue() {
    std::lock_guard<std::mutex> lock(imu_queue_mutex);
    if(keyframes.empty() || imu_queue.empty() || base_frame_id.empty()) {
      return false;
    }

    bool updated = false;
    auto imu_cursor = imu_queue.begin();

    for(auto& keyframe : keyframes) {
      if(keyframe->stamp > imu_queue.back()->header.stamp) {
        break;
      }

      if(keyframe->stamp < (*imu_cursor)->header.stamp || keyframe->acceleration) {
        continue;
      }

      // find imu data which is closest to the keyframe
      auto closest_imu = imu_cursor;
      for(auto imu = imu_cursor; imu != imu_queue.end(); imu++) {
        auto dt = (rclcpp::Time((*closest_imu)->header.stamp) - keyframe->stamp).seconds();
        auto dt2 = (rclcpp::Time((*imu)->header.stamp) - keyframe->stamp).seconds();
        if(std::abs(dt) < std::abs(dt2)) {
          break;
        }

        closest_imu = imu;
      }

      imu_cursor = closest_imu;
      if(0.2 < std::abs((rclcpp::Time((*closest_imu)->header.stamp) - keyframe->stamp).seconds())) {
        continue;
      }

      const auto& imu_ori = (*closest_imu)->orientation;
      const auto& imu_acc = (*closest_imu)->linear_acceleration;

      geometry_msgs::msg::Vector3Stamped acc_imu;
      geometry_msgs::msg::Vector3Stamped acc_base;
      geometry_msgs::msg::QuaternionStamped quat_imu;
      geometry_msgs::msg::QuaternionStamped quat_base;

      quat_imu.header.frame_id = acc_imu.header.frame_id = (*closest_imu)->header.frame_id;
      quat_imu.header.stamp = acc_imu.header.stamp = rclcpp::Time(0);
      acc_imu.vector = (*closest_imu)->linear_acceleration;
      quat_imu.quaternion = (*closest_imu)->orientation;

      geometry_msgs::msg::TransformStamped transform;
      try {
        transform = tf_buffer->lookupTransform(base_frame_id, (*closest_imu)->header.frame_id, rclcpp::Time(0));
      } catch(std::exception& e) {
        std::cerr << "failed to find transform!!" << std::endl;
        return false;
      }

      tf2::Transform transform_tf;
      transform_tf.setOrigin(tf2::Vector3(transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z));
      transform_tf.setRotation(tf2::Quaternion(transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z, transform.transform.rotation.w));
      tf2::Vector3 imu_frame_vect(acc_imu.vector.x, acc_imu.vector.y, acc_imu.vector.z);
      tf2::Quaternion imu_frame_quat(quat_imu.quaternion.x, quat_imu.quaternion.y, quat_imu.quaternion.z, quat_imu.quaternion.w);

      tf2::Vector3 base_frame_vect = transform_tf * imu_frame_vect;
      acc_base.header.frame_id = base_frame_id;
      acc_base.header.stamp = acc_imu.header.stamp;
      acc_base.vector.x = base_frame_vect.x();
      acc_base.vector.y = base_frame_vect.y();
      acc_base.vector.z = base_frame_vect.z();

      tf2::Quaternion base_frame_quat = transform_tf * imu_frame_quat;
      quat_base.header.frame_id = base_frame_id;
      quat_base.header.stamp = quat_imu.header.stamp;
      quat_base.quaternion.x = base_frame_quat.x();
      quat_base.quaternion.y = base_frame_quat.y();
      quat_base.quaternion.z = base_frame_quat.z();
      quat_base.quaternion.w = base_frame_quat.w();

      keyframe->acceleration = Eigen::Vector3d(acc_base.vector.x, acc_base.vector.y, acc_base.vector.z);
      keyframe->orientation = Eigen::Quaterniond(quat_base.quaternion.w, quat_base.quaternion.x, quat_base.quaternion.y, quat_base.quaternion.z);
      keyframe->orientation = keyframe->orientation;
      if(keyframe->orientation->w() < 0.0) {
        keyframe->orientation->coeffs() = -keyframe->orientation->coeffs();
      }

      if(params.enable_imu_orientation) {
        Eigen::MatrixXd info = Eigen::MatrixXd::Identity(3, 3) / params.imu_orientation_edge_stddev;
        auto edge = graph_slam->add_se3_prior_quat_edge(keyframe->node, *keyframe->orientation, info);
        graph_slam->add_robust_kernel(edge, params.imu_orientation_edge_robust_kernel, params.imu_orientation_edge_robust_kernel_size);
      }

      if(params.enable_imu_acceleration) {
        Eigen::MatrixXd info = Eigen::MatrixXd::Identity(3, 3) / params.imu_acceleration_edge_stddev;
        g2o::OptimizableGraph::Edge* edge = graph_slam->add_se3_prior_vec_edge(keyframe->node, -Eigen::Vector3d::UnitZ(), *keyframe->acceleration, info);
        graph_slam->add_robust_kernel(edge, params.imu_acceleration_edge_robust_kernel, params.imu_acceleration_edge_robust_kernel_size);
      }
      updated = true;
    }

    auto remove_loc = std::upper_bound(imu_queue.begin(), imu_queue.end(), keyframes.back()->stamp, [=](const rclcpp::Time& stamp, const sensor_msgs::msg::Imu::ConstSharedPtr& imu) { return stamp < rclcpp::Time(imu->header.stamp); });
    imu_queue.erase(imu_queue.begin(), remove_loc);

    return updated;
  }

  /**
   * @brief received floor coefficients are added to #floor_coeffs_queue
   * @param floor_coeffs_msg
   */
  void floor_coeffs_callback(const hdl_graph_slam::msg::FloorCoeffs::ConstSharedPtr floor_coeffs_msg) {
    if(floor_coeffs_msg->coeffs.empty()) {
      return;
    }

    std::lock_guard<std::mutex> lock(floor_coeffs_queue_mutex);
    floor_coeffs_queue.push_back(floor_coeffs_msg);
  }

  /**
   * @brief this methods associates floor coefficients messages with registered keyframes, and then adds the associated coeffs to the pose graph
   * @return if true, at least one floor plane edge is added to the pose graph
   */
  bool flush_floor_queue() {
    std::lock_guard<std::mutex> lock(floor_coeffs_queue_mutex);

    if(keyframes.empty()) {
      return false;
    }

    const auto& latest_keyframe_stamp = keyframes.back()->stamp;

    bool updated = false;
    for(const auto& floor_coeffs : floor_coeffs_queue) {
      if(rclcpp::Time(floor_coeffs->header.stamp) > latest_keyframe_stamp) {
        break;
      }

      auto found = keyframe_hash.find(floor_coeffs->header.stamp);
      if(found == keyframe_hash.end()) {
        continue;
      }

      if(!floor_plane_node) {
        floor_plane_node = graph_slam->add_plane_node(Eigen::Vector4d(0.0, 0.0, 1.0, 0.0));
        floor_plane_node->setFixed(true);
      }

      const auto& keyframe = found->second;

      Eigen::Vector4d coeffs(floor_coeffs->coeffs[0], floor_coeffs->coeffs[1], floor_coeffs->coeffs[2], floor_coeffs->coeffs[3]);
      Eigen::Matrix3d information = Eigen::Matrix3d::Identity() * (1.0 / params.floor_edge_stddev);
      auto edge = graph_slam->add_se3_plane_edge(keyframe->node, floor_plane_node, coeffs, information);
      graph_slam->add_robust_kernel(edge, params.floor_edge_robust_kernel, params.floor_edge_robust_kernel_size);

      keyframe->floor_coeffs = coeffs;

      updated = true;
    }

    auto remove_loc = std::upper_bound(floor_coeffs_queue.begin(), floor_coeffs_queue.end(), latest_keyframe_stamp, [=](const rclcpp::Time& stamp, const hdl_graph_slam::msg::FloorCoeffs::ConstSharedPtr coeffs) { return stamp < rclcpp::Time(coeffs->header.stamp); });
    floor_coeffs_queue.erase(floor_coeffs_queue.begin(), remove_loc);

    return updated;
  }

  /**
   * @brief generate map point cloud and publish it
   * @param event
   */
  void map_points_publish_timer_callback() {
    if(!map_points_pub->get_subscription_count() || !graph_updated) {
      return;
    }

    std::vector<KeyFrameSnapshot::Ptr> snapshot;

    keyframes_snapshot_mutex.lock();
    snapshot = keyframes_snapshot;
    keyframes_snapshot_mutex.unlock();

    auto cloud = map_cloud_generator->generate(snapshot, params.map_cloud_resolution);
    if(!cloud) {
      return;
    }

    cloud->header.frame_id = params.map_frame_id;
    cloud->header.stamp = snapshot.back()->cloud->header.stamp;

    sensor_msgs::msg::PointCloud2 cloud_msg;
    pcl::toROSMsg(*cloud, cloud_msg);
    map_points_pub->publish(cloud_msg);
  }

  /**
   * @brief this methods adds all the data in the queues to the pose graph, and then optimizes the pose graph
   * @param event
   */
  void optimization_timer_callback() {
    std::lock_guard<std::mutex> lock(main_thread_mutex);

    // add keyframes and floor coeffs in the queues to the pose graph
    bool keyframe_updated = flush_keyframe_queue();

    if(!keyframe_updated) {
      std_msgs::msg::Header read_until;
      read_until.stamp = get_clock()->now() + rclcpp::Duration(30, 0);
      read_until.frame_id = points_topic;
      read_until_pub->publish(read_until);
      read_until.frame_id = "/filtered_points";
      read_until_pub->publish(read_until);
    }

    if(!keyframe_updated && !flush_floor_queue() && !flush_gps_queue() && !flush_imu_queue()) {
      return;
    }

    // loop detection
    std::vector<Loop::Ptr> loops = loop_detector->detect(keyframes, new_keyframes, *graph_slam);
    for(const auto& loop : loops) {
      Eigen::Isometry3d relpose(loop->relative_pose.cast<double>());
      Eigen::MatrixXd information_matrix = inf_calclator->calc_information_matrix(loop->key1->cloud, loop->key2->cloud, relpose);
      auto edge = graph_slam->add_se3_edge(loop->key1->node, loop->key2->node, relpose, information_matrix);
      graph_slam->add_robust_kernel(edge, params.loop_closure_edge_robust_kernel, params.loop_closure_edge_robust_kernel_size);
    }
    std::copy(new_keyframes.begin(), new_keyframes.end(), std::back_inserter(keyframes));
    new_keyframes.clear();

    // move the first node anchor position to the current estimate of the first node pose
    // so the first node moves freely while trying to stay around the origin
    if(anchor_node && params.fix_first_node_adaptive){
      Eigen::Isometry3d anchor_target = static_cast<g2o::VertexSE3*>(anchor_edge->vertices()[1])->estimate();
      anchor_node->setEstimate(anchor_target);
    }
    // optimize the pose graph
    graph_slam->optimize(params.g2o_solver_num_iterations);
    if (keyframes.empty()) {
      return;
    }
    // publish tf
    const auto& keyframe = keyframes.back();
    Eigen::Isometry3d trans = keyframe->node->estimate() * keyframe->odom.inverse();
    trans_odom2map_mutex.lock();
    trans_odom2map = trans.matrix().cast<float>();
    trans_odom2map_mutex.unlock();

    std::vector<KeyFrameSnapshot::Ptr> snapshot(keyframes.size());
    std::transform(keyframes.begin(), keyframes.end(), snapshot.begin(), [=](const KeyFrame::Ptr& k) { return std::make_shared<KeyFrameSnapshot>(k); });

    keyframes_snapshot_mutex.lock();
    keyframes_snapshot.swap(snapshot);
    keyframes_snapshot_mutex.unlock();
    graph_updated = true;
    if(odom2map_pub->get_subscription_count()) {
      geometry_msgs::msg::TransformStamped ts = matrix2transform(keyframe->stamp, trans.matrix().cast<float>(), params.map_frame_id, params.odom_frame_id);
      odom2map_pub->publish(ts);
    }

    if(markers_pub->get_subscription_count()) {
      auto markers = create_marker_array(get_clock()->now());
      markers_pub->publish(markers);
    }
  }

  /**
   * @brief create visualization marker
   * @param stamp
   * @return
   */
  visualization_msgs::msg::MarkerArray create_marker_array(const rclcpp::Time& stamp) const {
    visualization_msgs::msg::MarkerArray markers;
    markers.markers.resize(4);

    // node markers
    visualization_msgs::msg::Marker& traj_marker = markers.markers[0];
    traj_marker.header.frame_id = "map";
    traj_marker.header.stamp = stamp;
    traj_marker.ns = "nodes";
    traj_marker.id = 0;
    traj_marker.type = visualization_msgs::msg::Marker::SPHERE_LIST;

    traj_marker.pose.orientation.w = 1.0;
    traj_marker.scale.x = traj_marker.scale.y = traj_marker.scale.z = 0.5;

    visualization_msgs::msg::Marker& imu_marker = markers.markers[1];
    imu_marker.header = traj_marker.header;
    imu_marker.ns = "imu";
    imu_marker.id = 1;
    imu_marker.type = visualization_msgs::msg::Marker::SPHERE_LIST;

    imu_marker.pose.orientation.w = 1.0;
    imu_marker.scale.x = imu_marker.scale.y = imu_marker.scale.z = 0.75;

    traj_marker.points.resize(keyframes.size());
    traj_marker.colors.resize(keyframes.size());
    for(int i = 0; i < keyframes.size(); i++) {
      Eigen::Vector3d pos = keyframes[i]->node->estimate().translation();
      traj_marker.points[i].x = pos.x();
      traj_marker.points[i].y = pos.y();
      traj_marker.points[i].z = pos.z();

      double p = static_cast<double>(i) / keyframes.size();
      traj_marker.colors[i].r = 1.0 - p;
      traj_marker.colors[i].g = p;
      traj_marker.colors[i].b = 0.0;
      traj_marker.colors[i].a = 1.0;

      if(keyframes[i]->acceleration) {
        Eigen::Vector3d pos = keyframes[i]->node->estimate().translation();
        geometry_msgs::msg::Point point;
        point.x = pos.x();
        point.y = pos.y();
        point.z = pos.z();

        std_msgs::msg::ColorRGBA color;
        color.r = 0.0;
        color.g = 0.0;
        color.b = 1.0;
        color.a = 0.1;

        imu_marker.points.push_back(point);
        imu_marker.colors.push_back(color);
      }
    }

    // edge markers
    visualization_msgs::msg::Marker& edge_marker = markers.markers[2];
    edge_marker.header.frame_id = "map";
    edge_marker.header.stamp = stamp;
    edge_marker.ns = "edges";
    edge_marker.id = 2;
    edge_marker.type = visualization_msgs::msg::Marker::LINE_LIST;

    edge_marker.pose.orientation.w = 1.0;
    edge_marker.scale.x = 0.05;

    edge_marker.points.resize(graph_slam->graph->edges().size() * 2);
    edge_marker.colors.resize(graph_slam->graph->edges().size() * 2);

    auto edge_itr = graph_slam->graph->edges().begin();
    for(int i = 0; edge_itr != graph_slam->graph->edges().end(); edge_itr++, i++) {
      g2o::HyperGraph::Edge* edge = *edge_itr;
      g2o::EdgeSE3* edge_se3 = dynamic_cast<g2o::EdgeSE3*>(edge);
      if(edge_se3) {
        g2o::VertexSE3* v1 = dynamic_cast<g2o::VertexSE3*>(edge_se3->vertices()[0]);
        g2o::VertexSE3* v2 = dynamic_cast<g2o::VertexSE3*>(edge_se3->vertices()[1]);
        Eigen::Vector3d pt1 = v1->estimate().translation();
        Eigen::Vector3d pt2 = v2->estimate().translation();

        edge_marker.points[i * 2].x = pt1.x();
        edge_marker.points[i * 2].y = pt1.y();
        edge_marker.points[i * 2].z = pt1.z();
        edge_marker.points[i * 2 + 1].x = pt2.x();
        edge_marker.points[i * 2 + 1].y = pt2.y();
        edge_marker.points[i * 2 + 1].z = pt2.z();

        double p1 = static_cast<double>(v1->id()) / graph_slam->graph->vertices().size();
        double p2 = static_cast<double>(v2->id()) / graph_slam->graph->vertices().size();
        edge_marker.colors[i * 2].r = 1.0 - p1;
        edge_marker.colors[i * 2].g = p1;
        edge_marker.colors[i * 2].a = 1.0;
        edge_marker.colors[i * 2 + 1].r = 1.0 - p2;
        edge_marker.colors[i * 2 + 1].g = p2;
        edge_marker.colors[i * 2 + 1].a = 1.0;

        if(std::abs(v1->id() - v2->id()) > 2) {
          edge_marker.points[i * 2].z += 0.5;
          edge_marker.points[i * 2 + 1].z += 0.5;
        }

        continue;
      }

      g2o::EdgeSE3Plane* edge_plane = dynamic_cast<g2o::EdgeSE3Plane*>(edge);
      if(edge_plane) {
        g2o::VertexSE3* v1 = dynamic_cast<g2o::VertexSE3*>(edge_plane->vertices()[0]);
        Eigen::Vector3d pt1 = v1->estimate().translation();
        Eigen::Vector3d pt2(pt1.x(), pt1.y(), 0.0);

        edge_marker.points[i * 2].x = pt1.x();
        edge_marker.points[i * 2].y = pt1.y();
        edge_marker.points[i * 2].z = pt1.z();
        edge_marker.points[i * 2 + 1].x = pt2.x();
        edge_marker.points[i * 2 + 1].y = pt2.y();
        edge_marker.points[i * 2 + 1].z = pt2.z();

        edge_marker.colors[i * 2].b = 1.0;
        edge_marker.colors[i * 2].a = 1.0;
        edge_marker.colors[i * 2 + 1].b = 1.0;
        edge_marker.colors[i * 2 + 1].a = 1.0;

        continue;
      }

      g2o::EdgeSE3PriorXY* edge_priori_xy = dynamic_cast<g2o::EdgeSE3PriorXY*>(edge);
      if(edge_priori_xy) {
        g2o::VertexSE3* v1 = dynamic_cast<g2o::VertexSE3*>(edge_priori_xy->vertices()[0]);
        Eigen::Vector3d pt1 = v1->estimate().translation();
        Eigen::Vector3d pt2 = Eigen::Vector3d::Zero();
        pt2.head<2>() = edge_priori_xy->measurement();

        edge_marker.points[i * 2].x = pt1.x();
        edge_marker.points[i * 2].y = pt1.y();
        edge_marker.points[i * 2].z = pt1.z() + 0.5;
        edge_marker.points[i * 2 + 1].x = pt2.x();
        edge_marker.points[i * 2 + 1].y = pt2.y();
        edge_marker.points[i * 2 + 1].z = pt2.z() + 0.5;

        edge_marker.colors[i * 2].r = 1.0;
        edge_marker.colors[i * 2].a = 1.0;
        edge_marker.colors[i * 2 + 1].r = 1.0;
        edge_marker.colors[i * 2 + 1].a = 1.0;

        continue;
      }

      g2o::EdgeSE3PriorXYZ* edge_priori_xyz = dynamic_cast<g2o::EdgeSE3PriorXYZ*>(edge);
      if(edge_priori_xyz) {
        g2o::VertexSE3* v1 = dynamic_cast<g2o::VertexSE3*>(edge_priori_xyz->vertices()[0]);
        Eigen::Vector3d pt1 = v1->estimate().translation();
        Eigen::Vector3d pt2 = edge_priori_xyz->measurement();

        edge_marker.points[i * 2].x = pt1.x();
        edge_marker.points[i * 2].y = pt1.y();
        edge_marker.points[i * 2].z = pt1.z() + 0.5;
        edge_marker.points[i * 2 + 1].x = pt2.x();
        edge_marker.points[i * 2 + 1].y = pt2.y();
        edge_marker.points[i * 2 + 1].z = pt2.z();

        edge_marker.colors[i * 2].r = 1.0;
        edge_marker.colors[i * 2].a = 1.0;
        edge_marker.colors[i * 2 + 1].r = 1.0;
        edge_marker.colors[i * 2 + 1].a = 1.0;

        continue;
      }
    }

    // sphere
    visualization_msgs::msg::Marker& sphere_marker = markers.markers[3];
    sphere_marker.header.frame_id = "map";
    sphere_marker.header.stamp = stamp;
    sphere_marker.ns = "loop_close_radius";
    sphere_marker.id = 3;
    sphere_marker.type = visualization_msgs::msg::Marker::SPHERE;

    if(!keyframes.empty()) {
      Eigen::Vector3d pos = keyframes.back()->node->estimate().translation();
      sphere_marker.pose.position.x = pos.x();
      sphere_marker.pose.position.y = pos.y();
      sphere_marker.pose.position.z = pos.z();
    }
    sphere_marker.pose.orientation.w = 1.0;
    sphere_marker.scale.x = sphere_marker.scale.y = sphere_marker.scale.z = loop_detector->get_distance_thresh() * 2.0;

    sphere_marker.color.r = 1.0;
    sphere_marker.color.a = 0.3;

    return markers;
  }

  /**
   * @brief dump all data to the current directory
   * @param req
   * @param res
   * @return
   */
  bool dump_service(std::shared_ptr<hdl_graph_slam::srv::DumpGraph::Request> req, std::shared_ptr<hdl_graph_slam::srv::DumpGraph::Response> res) {
    std::lock_guard<std::mutex> lock(main_thread_mutex);

    std::string directory = req->destination;

    if(directory.empty()) {
      std::array<char, 64> buffer;
      buffer.fill(0);
      time_t rawtime;
      time(&rawtime);
      const auto timeinfo = localtime(&rawtime);
      strftime(buffer.data(), sizeof(buffer), "%d-%m-%Y %H:%M:%S", timeinfo);
    }

    if(!boost::filesystem::is_directory(directory)) {
      boost::filesystem::create_directory(directory);
    }

    std::cout << "all data dumped to:" << directory << std::endl;

    graph_slam->save(directory + "/graph.g2o");
    for(int i = 0; i < keyframes.size(); i++) {
      std::stringstream sst;
      sst << boost::format("%s/%06d") % directory % i;

      keyframes[i]->save(sst.str());
    }

    if(zero_utm) {
      std::ofstream zero_utm_ofs(directory + "/zero_utm");
      zero_utm_ofs << boost::format("%.6f %.6f %.6f") % zero_utm->x() % zero_utm->y() % zero_utm->z() << std::endl;
    }

    std::ofstream ofs(directory + "/special_nodes.csv");
    ofs << "anchor_node " << (anchor_node == nullptr ? -1 : anchor_node->id()) << std::endl;
    ofs << "anchor_edge " << (anchor_edge == nullptr ? -1 : anchor_edge->id()) << std::endl;
    ofs << "floor_node " << (floor_plane_node == nullptr ? -1 : floor_plane_node->id()) << std::endl;

    res->success = true;
    return true;
  }

  /**
   * @brief save map data as pcd
   * @param req
   * @param res
   * @return
   */
  bool save_map_service(std::shared_ptr<hdl_graph_slam::srv::SaveMap::Request> req, std::shared_ptr<hdl_graph_slam::srv::SaveMap::Response> res) {
    std::vector<KeyFrameSnapshot::Ptr> snapshot;

    keyframes_snapshot_mutex.lock();
    snapshot = keyframes_snapshot;
    keyframes_snapshot_mutex.unlock();

    auto cloud = map_cloud_generator->generate(snapshot, req->resolution);
    if(!cloud) {
      res->success = false;
      return true;
    }

    if(zero_utm && req->utm) {
      for(auto& pt : cloud->points) {
        pt.getVector3fMap() += (*zero_utm).cast<float>();
      }
    }

    cloud->header.frame_id = params.map_frame_id;
    cloud->header.stamp = snapshot.back()->cloud->header.stamp;

    if(zero_utm) {
      std::ofstream ofs(req->destination + ".utm");
      ofs << boost::format("%.6f %.6f %.6f") % zero_utm->x() % zero_utm->y() % zero_utm->z() << std::endl;
    }

    int ret = pcl::io::savePCDFileBinary(req->destination, *cloud);
    res->success = ret == 0;

    return true;
  }

private:
  // ROS
  rclcpp::TimerBase::SharedPtr optimization_timer;
  rclcpp::TimerBase::SharedPtr map_publish_timer;

  std::unique_ptr<message_filters::Subscriber<nav_msgs::msg::Odometry>> odom_sub;
  std::unique_ptr<message_filters::Subscriber<sensor_msgs::msg::PointCloud2>> cloud_sub;
  std::unique_ptr<message_filters::Synchronizer<ApproxSyncPolicy>> sync;

  rclcpp::Subscription<geographic_msgs::msg::GeoPointStamped>::SharedPtr gps_sub;
  rclcpp::Subscription<nmea_msgs::msg::Sentence>::SharedPtr nmea_sub;
  rclcpp::Subscription<sensor_msgs::msg::NavSatFix>::SharedPtr navsat_sub;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub;
  rclcpp::Subscription<msg::FloorCoeffs>::SharedPtr floor_sub;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr markers_pub;

  std::mutex trans_odom2map_mutex;
  Eigen::Matrix4f trans_odom2map;
  rclcpp::Publisher<geometry_msgs::msg::TransformStamped>::SharedPtr odom2map_pub;

  std::string points_topic;
  rclcpp::Publisher<std_msgs::msg::Header>::SharedPtr read_until_pub;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr map_points_pub;

  std::shared_ptr<tf2_ros::TransformListener> tf_listener;
  std::unique_ptr<tf2_ros::Buffer> tf_buffer;

  rclcpp::Service<srv::DumpGraph>::SharedPtr dump_service_server;
  rclcpp::Service<srv::SaveMap>::SharedPtr save_map_service_server;

  // keyframe queue
  std::string base_frame_id;
  std::mutex keyframe_queue_mutex;
  std::deque<KeyFrame::Ptr> keyframe_queue;

  // gps queue
  boost::optional<Eigen::Vector3d> zero_utm;
  std::mutex gps_queue_mutex;
  std::deque<geographic_msgs::msg::GeoPointStamped::ConstSharedPtr> gps_queue;

  // imu queue
  std::mutex imu_queue_mutex;
  std::deque<sensor_msgs::msg::Imu::ConstSharedPtr> imu_queue;

  // floor_coeffs queue
  std::mutex floor_coeffs_queue_mutex;
  std::deque<hdl_graph_slam::msg::FloorCoeffs::ConstSharedPtr> floor_coeffs_queue;

  // for map cloud generation
  std::atomic_bool graph_updated;
  std::mutex keyframes_snapshot_mutex;
  std::vector<KeyFrameSnapshot::Ptr> keyframes_snapshot;
  std::unique_ptr<MapCloudGenerator> map_cloud_generator;

  // graph slam
  // all the below members must be accessed after locking main_thread_mutex
  std::mutex main_thread_mutex;

  std::deque<KeyFrame::Ptr> new_keyframes;

  g2o::VertexSE3* anchor_node;
  g2o::EdgeSE3* anchor_edge;
  g2o::VertexPlane* floor_plane_node;
  std::vector<KeyFrame::Ptr> keyframes;
  std::unordered_map<rclcpp::Time, KeyFrame::Ptr, RosTimeHash> keyframe_hash;

  std::unique_ptr<GraphSLAM> graph_slam;
  std::unique_ptr<LoopDetector> loop_detector;
  std::unique_ptr<KeyframeUpdater> keyframe_updater;
  std::unique_ptr<NmeaSentenceParser> nmea_parser;

  std::unique_ptr<InformationMatrixCalculator> inf_calclator;
};

}  // namespace hdl_graph_slam

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(hdl_graph_slam::HdlGraphSlamNode)
