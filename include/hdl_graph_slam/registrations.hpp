// SPDX-License-Identifier: BSD-2-Clause

#ifndef HDL_GRAPH_SLAM_REGISTRATIONS_HPP
#define HDL_GRAPH_SLAM_REGISTRATIONS_HPP

#include <pcl/registration/registration.h>

namespace hdl_graph_slam {

typedef struct {
  std::string registration_method;
  std::string reg_nn_search_method;
  double reg_transformation_epsilon;
  double reg_max_correspondence_distance;
  double reg_resolution;
  int reg_num_threads;
  int reg_maximum_iterations;
  int reg_correspondence_randomness;
  int reg_max_optimizer_iterations;
  bool reg_use_reciprocal_correspondences;
} RegistrationParams;

pcl::Registration<pcl::PointXYZI, pcl::PointXYZI>::Ptr select_registration_method(const RegistrationParams& params);

}  // namespace hdl_graph_slam

#endif  //
