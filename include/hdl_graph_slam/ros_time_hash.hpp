// SPDX-License-Identifier: BSD-2-Clause

#ifndef ROS_TIME_HASH_HPP
#define ROS_TIME_HASH_HPP

#include <unordered_map>
#include <boost/functional/hash.hpp>

#include <rclcpp/time.hpp>

/**
 * @brief Hash calculation for ros::Time
 */
class RosTimeHash {
public:
  size_t operator()(const rclcpp::Time& val) const {
    size_t seed = 0;
    uint64_t sec = val.nanoseconds() / 1000000000;
    uint64_t nsec = val.nanoseconds() - sec * 1000000000;
    boost::hash_combine(seed, sec);
    boost::hash_combine(seed, nsec);
    return seed;
  }
};

#endif  // ROS_TIME_HASHER_HPP
