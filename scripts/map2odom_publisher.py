#!/usr/bin/python3
# SPDX-License-Identifier: BSD-2-Clause
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped


class Map2OdomPublisher(Node):
	def __init__(self):
		super().__init__('map2odom_publisher')
		self.broadcaster = TransformBroadcaster(self)
		self.subscriber = self.create_subscription(TransformStamped, '/hdl_graph_slam/odom2pub', self.callback, 1)
		self.timer = self.create_timer(0.1, self.spin)
		self.odom_msg = TransformStamped()
		self.odom_msg.header.frame_id = 'map'
		self.odom_msg.child_frame_id = 'odom'
		self.odom_msg.transform.rotation.w = 1.0

	def callback(self, odom_msg):
		self.odom_msg = odom_msg

	def spin(self):
		if self.odom_msg.header.stamp == Time():
			return

		self.broadcaster.sendTransform(self.odom_msg)


if __name__ == '__main__':
	rclpy.init()
	node = Map2OdomPublisher()
	rclpy.spin(node)
	node.destroy_node()
	rclpy.shutdown()
