# SPDX-License-Identifier: BSD-2-Clause
cmake_minimum_required(VERSION 3.5)
project(hdl_graph_slam)

# Can we use C++17 in indigo?
add_definitions(-msse -msse2 -msse3 -msse4 -msse4.1 -msse4.2)
set(CMAKE_CXX_FLAGS "-msse -msse2 -msse3 -msse4 -msse4.1 -msse4.2")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_LIST_DIR}/cmake")

# pcl 1.7 causes a segfault when it is built with debug mode
set(CMAKE_BUILD_TYPE "RELEASE")

# find_package(catkin REQUIRED COMPONENTS
#   roscpp
#   rospy
#   pcl_ros
#   geodesy
#   nmea_msgs
#   sensor_msgs
#   geometry_msgs
#   message_generation
#   interactive_markers
#   ndt_omp
#   fast_gicp
# )
# catkin_python_setup()
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(geographic_msgs REQUIRED)
find_package(nmea_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(interactive_markers REQUIRED)
find_package(fast_gicp REQUIRED)
find_package(ndt_omp REQUIRED)
find_package(nav_msgs REQUIRED)

find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

message(STATUS "PCL_INCLUDE_DIRS:" ${PCL_INCLUDE_DIRS})
message(STATUS "PCL_LIBRARY_DIRS:" ${PCL_LIBRARY_DIRS})
message(STATUS "PCL_DEFINITIONS:" ${PCL_DEFINITIONS})

find_package(G2O REQUIRED)
include_directories(SYSTEM ${G2O_INCLUDE_DIR} ${G2O_INCLUDE_DIRS})
link_directories(${G2O_LIBRARY_DIRS})
# link_libraries(${G2O_LIBRARIES})

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

find_library(VGICP_CUDA_FOUND NAMES fast_vgicp_cuda)
message(STATUS "VGICP_CUDA_FOUND:" ${VGICP_CUDA_FOUND})
if(VGICP_CUDA_FOUND)
  add_definitions(-DUSE_VGICP_CUDA)
endif()

rosidl_generate_interfaces(${PROJECT_NAME}
  msg/FloorCoeffs.msg
  msg/ScanMatchingStatus.msg
  srv/SaveMap.srv
  srv/DumpGraph.srv
DEPENDENCIES
  std_msgs
  geometry_msgs
)
ament_export_dependencies(rosidl_default_runtime)

add_library(${PROJECT_NAME}_lib SHARED
  apps/prefiltering_node.cpp
  apps/scan_matching_odometry_node.cpp
  apps/floor_detection_node.cpp
  apps/hdl_graph_slam_node.cpp
  src/hdl_graph_slam/registrations.cpp
  src/hdl_graph_slam/keyframe.cpp
  src/hdl_graph_slam/map_cloud_generator.cpp
  src/hdl_graph_slam/information_matrix_calculator.cpp
  src/g2o/robust_kernel_io.cpp
  src/hdl_graph_slam/graph_slam.cpp
)
ament_target_dependencies(${PROJECT_NAME}_lib
  rclcpp
  pcl_ros
  geometry_msgs
  geographic_msgs
  nmea_msgs
  sensor_msgs
  interactive_markers
  fast_gicp
  ndt_omp
  nav_msgs
)
rosidl_target_interfaces(${PROJECT_NAME}_lib
  ${PROJECT_NAME}
  rosidl_typesupport_cpp
)
target_include_directories(
  ${PROJECT_NAME}_lib PUBLIC
  ${PCL_INCLUDE_DIRS}
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(${PROJECT_NAME}_lib
  ${PCL_LIBRARIES}
  ${G2O_TYPES_DATA}
  ${G2O_CORE_LIBRARY}
  ${G2O_STUFF_LIBRARY}
  ${G2O_SOLVER_PCG}
  ${G2O_SOLVER_CSPARSE}   # be aware of that CSPARSE is released under LGPL
  ${G2O_SOLVER_CHOLMOD}   # be aware of that cholmod is released under GPL
  ${G2O_TYPES_SLAM3D}
  ${G2O_TYPES_SLAM3D_ADDONS}
)
rclcpp_components_register_node(${PROJECT_NAME}_lib
  PLUGIN "${PROJECT_NAME}::PrefilteringNode"
  EXECUTABLE ${PROJECT_NAME}_prefiltering_node
)
rclcpp_components_register_node(${PROJECT_NAME}_lib
  PLUGIN "${PROJECT_NAME}::ScanMatchingOdometryNode"
  EXECUTABLE ${PROJECT_NAME}_scan_matching_odometry_node
)
rclcpp_components_register_node(${PROJECT_NAME}_lib
  PLUGIN "${PROJECT_NAME}::FloorDetectionNode"
  EXECUTABLE ${PROJECT_NAME}_floor_detection_node
)
rclcpp_components_register_node(${PROJECT_NAME}_lib
  PLUGIN "${PROJECT_NAME}::HdlGraphSlamNode"
  EXECUTABLE ${PROJECT_NAME}_node
)

install(
  TARGETS
    ${PROJECT_NAME}_lib
    ${PROJECT_NAME}_prefiltering_node
    ${PROJECT_NAME}_scan_matching_odometry_node
    ${PROJECT_NAME}_floor_detection_node
    ${PROJECT_NAME}_node
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
)

install(
  PROGRAMS
    scripts/map2odom_publisher.py
  DESTINATION
    lib/${PROJECT_NAME}
)

ament_package()

# nodelets
# add_library(prefiltering_nodelet apps/prefiltering_nodelet.cpp)
# target_link_libraries(prefiltering_nodelet
#   ${catkin_LIBRARIES}
#   ${PCL_LIBRARIES}
# )


# add_library(floor_detection_nodelet apps/floor_detection_nodelet.cpp)
# target_link_libraries(floor_detection_nodelet
#   ${catkin_LIBRARIES}
#   ${PCL_LIBRARIES}
# )
# add_dependencies(floor_detection_nodelet ${PROJECT_NAME}_gencpp)


# add_library(scan_matching_odometry_nodelet
#   apps/scan_matching_odometry_nodelet.cpp
#   src/hdl_graph_slam/registrations.cpp
# )
# target_link_libraries(scan_matching_odometry_nodelet
#   ${catkin_LIBRARIES}
#   ${PCL_LIBRARIES}
# )
# add_dependencies(scan_matching_odometry_nodelet ${PROJECT_NAME}_gencpp)


# add_library(hdl_graph_slam_nodelet
#   apps/hdl_graph_slam_nodelet.cpp
#   src/hdl_graph_slam/graph_slam.cpp
#   src/hdl_graph_slam/keyframe.cpp
#   src/hdl_graph_slam/map_cloud_generator.cpp
#   src/hdl_graph_slam/registrations.cpp
#   src/hdl_graph_slam/information_matrix_calculator.cpp
#   src/g2o/robust_kernel_io.cpp
# )
# target_link_libraries(hdl_graph_slam_nodelet
#   ${catkin_LIBRARIES}
#   ${PCL_LIBRARIES}
#   ${G2O_TYPES_DATA}
#   ${G2O_CORE_LIBRARY}
#   ${G2O_STUFF_LIBRARY}
#   ${G2O_SOLVER_PCG}
#   ${G2O_SOLVER_CSPARSE}   # be aware of that CSPARSE is released under LGPL
#   ${G2O_SOLVER_CHOLMOD}   # be aware of that cholmod is released under GPL
#   ${G2O_TYPES_SLAM3D}
#   ${G2O_TYPES_SLAM3D_ADDONS}
# )
# add_dependencies(hdl_graph_slam_nodelet ${PROJECT_NAME}_gencpp)

# catkin_install_python(
#   PROGRAMS
#     src/${PROJECT_NAME}/bag_player.py
#     src/${PROJECT_NAME}/ford2bag.py
#     src/${PROJECT_NAME}/map2odom_publisher.py
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

# install(FILES nodelet_plugins.xml
#   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
# )

# install(TARGETS
#   prefiltering_nodelet
#   floor_detection_nodelet
#   scan_matching_odometry_nodelet
#   hdl_graph_slam_nodelet
#   LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION})
