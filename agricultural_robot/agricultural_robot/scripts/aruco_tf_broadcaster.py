#!/usr/bin/env python3
"""
Pubblica una TF per ogni marker rilevato da aruco_ros/marker_publisher.

Sottoscrive:  /aruco_marker_publisher/markers  (aruco_msgs/MarkerArray)
Pubblica TF:  <header.frame_id dei marker>  ->  aruco_<id>

Cosi' la posa di un marker e' disponibile nell'albero TF
(es. base_link -> aruco_201) e puo' essere usata come frame_id
dell'obiettivo del CLIK.
"""

import rclpy
from aruco_msgs.msg import MarkerArray
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from tf2_ros import TransformBroadcaster


class ArucoTfBroadcaster(Node):

    def __init__(self):
        super().__init__("aruco_tf_broadcaster")
        self.declare_parameter("markers_topic", "/aruco_marker_publisher/markers")
        self.declare_parameter("frame_prefix", "aruco_")

        topic = self.get_parameter("markers_topic").value
        self.prefix = self.get_parameter("frame_prefix").value

        self.broadcaster = TransformBroadcaster(self)
        self.create_subscription(MarkerArray, topic, self.markers_cb, 10)
        self.seen_ids = set()
        self.get_logger().info(f"Pubblico le TF dei marker da '{topic}' (prefisso '{self.prefix}')")

    def markers_cb(self, msg: MarkerArray):
        transforms = []
        for m in msg.markers:
            parent = m.header.frame_id or msg.header.frame_id
            if not parent:
                continue
            stamp = m.header.stamp
            if stamp.sec == 0 and stamp.nanosec == 0:
                stamp = self.get_clock().now().to_msg()

            t = TransformStamped()
            t.header.stamp = stamp
            t.header.frame_id = parent
            t.child_frame_id = f"{self.prefix}{m.id}"
            t.transform.translation.x = m.pose.pose.position.x
            t.transform.translation.y = m.pose.pose.position.y
            t.transform.translation.z = m.pose.pose.position.z
            t.transform.rotation = m.pose.pose.orientation
            transforms.append(t)

            if m.id not in self.seen_ids:
                self.seen_ids.add(m.id)
                self.get_logger().info(f"Nuovo marker: {t.child_frame_id} (padre '{parent}')")

        if transforms:
            self.broadcaster.sendTransform(transforms)


def main(args=None):
    rclpy.init(args=args)
    node = ArucoTfBroadcaster()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
