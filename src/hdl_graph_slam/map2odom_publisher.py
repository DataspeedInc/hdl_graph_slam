#!/usr/bin/python
# SPDX-License-Identifier: BSD-2-Clause
import tf
import rospy
from geometry_msgs.msg import *


class Map2OdomPublisher:
	def __init__(self):
		self.broadcaster = tf.TransformBroadcaster()
		self.subscriber = rospy.Subscriber('/hdl_graph_slam/odom2pub', TransformStamped, self.callback)
		self.timer = rospy.Timer(rospy.Duration(0.1), self.spin, oneshot=False, reset=True)

	def callback(self, odom_msg):
		self.odom_msg = odom_msg

	def spin(self, event):
		if not hasattr(self, 'odom_msg'):
			self.broadcaster.sendTransform((0, 0, 0), (0, 0, 0, 1), event.current_real, 'odom', 'map')
			return

		pose = self.odom_msg.transform
		pos = (pose.translation.x, pose.translation.y, pose.translation.z)
		quat = (pose.rotation.x, pose.rotation.y, pose.rotation.z, pose.rotation.w)

		map_frame_id = self.odom_msg.header.frame_id
		odom_frame_id = self.odom_msg.child_frame_id

		self.broadcaster.sendTransform(pos, quat, event.current_real, odom_frame_id, map_frame_id)


if __name__ == '__main__':
	rospy.init_node('map2odom_publisher')
	node = Map2OdomPublisher()
	rospy.spin()
