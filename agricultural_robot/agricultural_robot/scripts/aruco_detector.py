#!/usr/bin/env python3
"""
Nodo di rilevamento di marker ArUco multipli per agricultural_robot.

Sottoscrive:
  image_topic        (sensor_msgs/Image)       immagine della camera sinistra ZED
  camera_info_topic  (sensor_msgs/CameraInfo)  intrinseci (K, D) e frame ottico

Pubblica:
  /aruco/markers      (aruco_msgs/MarkerArray) posa di ogni marker nel frame ottico
  /aruco/debug_image  (sensor_msgs/Image)      immagine con contorni, ID e assi
  TF  <frame ottico> -> aruco_<id>             una trasformata per ogni marker

La posa e' stimata con solvePnP (metodo IPPE_SQUARE) marker per marker:
funziona sia con OpenCV 4.5 (Ubuntu 22.04 / Humble) sia con le versioni recenti.
"""

import math

import cv2
import numpy as np
import rclpy
from aruco_msgs.msg import Marker, MarkerArray
from cv_bridge import CvBridge
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import TransformBroadcaster


def rotation_matrix_to_quaternion(R):
    """Converte una matrice di rotazione 3x3 in quaternione (x, y, z, w)."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0.0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return x, y, z, w


class ArucoDetector(Node):

    def __init__(self):
        super().__init__("aruco_detector")

        # ---------------- Parametri ----------------
        self.declare_parameter("image_topic", "/zed/left/image")
        self.declare_parameter("camera_info_topic", "/zed/left/camera_info")
        self.declare_parameter("dictionary", "DICT_ARUCO_ORIGINAL")
        self.declare_parameter("marker_size", 0.12)        # lato del quadrato nero [m]
        self.declare_parameter("marker_ids", [-1])         # [-1] = accetta tutti gli ID
        self.declare_parameter("publish_tf", True)
        self.declare_parameter("publish_debug_image", True)
        self.declare_parameter("tf_prefix", "aruco_")

        image_topic = self.get_parameter("image_topic").value
        info_topic = self.get_parameter("camera_info_topic").value
        dict_name = self.get_parameter("dictionary").value
        self.marker_size = float(self.get_parameter("marker_size").value)
        ids = list(self.get_parameter("marker_ids").value)
        self.allowed_ids = None if (len(ids) == 0 or -1 in ids) else set(int(i) for i in ids)
        self.publish_tf = bool(self.get_parameter("publish_tf").value)
        self.publish_debug = bool(self.get_parameter("publish_debug_image").value)
        self.tf_prefix = self.get_parameter("tf_prefix").value

        # ---------------- Detector ArUco ----------------
        if not hasattr(cv2.aruco, dict_name):
            raise ValueError(f"Dizionario ArUco sconosciuto: {dict_name}")
        dict_id = getattr(cv2.aruco, dict_name)

        if hasattr(cv2.aruco, "ArucoDetector"):            # OpenCV >= 4.7
            self.dictionary = cv2.aruco.getPredefinedDictionary(dict_id)
            params = cv2.aruco.DetectorParameters()
            params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
            self.detector = cv2.aruco.ArucoDetector(self.dictionary, params)
            self._detect = lambda gray: self.detector.detectMarkers(gray)
        else:                                              # OpenCV 4.5 (Humble)
            self.dictionary = cv2.aruco.Dictionary_get(dict_id)
            params = cv2.aruco.DetectorParameters_create()
            params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
            self._detect = lambda gray: cv2.aruco.detectMarkers(
                gray, self.dictionary, parameters=params)

        # Punti 3D degli angoli nel frame del marker (ordine richiesto da IPPE_SQUARE):
        # alto-sx, alto-dx, basso-dx, basso-sx. Z esce dalla faccia del marker.
        h = self.marker_size / 2.0
        self.obj_points = np.array(
            [[-h, h, 0.0], [h, h, 0.0], [h, -h, 0.0], [-h, -h, 0.0]], dtype=np.float32)

        # ---------------- Stato camera ----------------
        self.K = None
        self.D = None
        self.camera_frame = None
        self.seen_ids = set()      # per il messaggio "nuovo marker" (una volta per ID)

        # ---------------- ROS I/O ----------------
        self.bridge = CvBridge()
        self.tf_broadcaster = TransformBroadcaster(self)
        self.markers_pub = self.create_publisher(MarkerArray, "/aruco/markers", 10)
        self.debug_pub = self.create_publisher(Image, "/aruco/debug_image", 1)

        self.create_subscription(CameraInfo, info_topic, self.camera_info_cb, 10)
        self.create_subscription(Image, image_topic, self.image_cb, qos_profile_sensor_data)

        self.get_logger().info(
            f"ArUco detector: {dict_name}, lato {self.marker_size:.3f} m, "
            f"immagine '{image_topic}', camera_info '{info_topic}'")

    # ------------------------------------------------------------------
    def camera_info_cb(self, msg: CameraInfo):
        if self.K is None:
            self.get_logger().info(f"camera_info ricevuta (frame '{msg.header.frame_id}')")
        self.K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        self.D = np.array(msg.d, dtype=np.float64) if len(msg.d) > 0 else np.zeros(5)
        self.camera_frame = msg.header.frame_id

    # ------------------------------------------------------------------
    def image_cb(self, msg: Image):
        if self.K is None:
            self.get_logger().warn("In attesa di camera_info...", throttle_duration_sec=5.0)
            return

        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self._detect(gray)

        frame_id = msg.header.frame_id or self.camera_frame
        out = MarkerArray()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = frame_id
        transforms = []

        if ids is not None:
            for marker_corners, marker_id in zip(corners, ids.flatten()):
                marker_id = int(marker_id)
                if self.allowed_ids is not None and marker_id not in self.allowed_ids:
                    continue

                img_points = marker_corners.reshape(4, 2).astype(np.float32)
                ok, rvec, tvec = cv2.solvePnP(
                    self.obj_points, img_points, self.K, self.D,
                    flags=cv2.SOLVEPNP_IPPE_SQUARE)
                if not ok:
                    continue

                R, _ = cv2.Rodrigues(rvec)
                qx, qy, qz, qw = rotation_matrix_to_quaternion(R)
                t = tvec.flatten()

                if marker_id not in self.seen_ids:
                    self.seen_ids.add(marker_id)
                    self.get_logger().info(
                        f"Nuovo marker: {self.tf_prefix}{marker_id} a {float(np.linalg.norm(t)):.2f} m "
                        f"(padre '{frame_id}')")

                m = Marker()
                m.header = out.header
                m.id = marker_id
                m.pose.pose.position.x = float(t[0])
                m.pose.pose.position.y = float(t[1])
                m.pose.pose.position.z = float(t[2])
                m.pose.pose.orientation.x = qx
                m.pose.pose.orientation.y = qy
                m.pose.pose.orientation.z = qz
                m.pose.pose.orientation.w = qw
                m.confidence = 1.0
                out.markers.append(m)

                if self.publish_tf:
                    tf = TransformStamped()
                    tf.header = out.header
                    tf.child_frame_id = f"{self.tf_prefix}{marker_id}"
                    tf.transform.translation.x = float(t[0])
                    tf.transform.translation.y = float(t[1])
                    tf.transform.translation.z = float(t[2])
                    tf.transform.rotation.x = qx
                    tf.transform.rotation.y = qy
                    tf.transform.rotation.z = qz
                    tf.transform.rotation.w = qw
                    transforms.append(tf)

                if self.publish_debug:
                    if hasattr(cv2, "drawFrameAxes"):
                        cv2.drawFrameAxes(frame, self.K, self.D, rvec, tvec, self.marker_size * 0.5)
                    else:                                  # OpenCV piu' vecchi
                        cv2.aruco.drawAxis(frame, self.K, self.D, rvec, tvec, self.marker_size * 0.5)
                    proj, _ = cv2.projectPoints(self.obj_points, rvec, tvec, self.K, self.D)
                    reproj = float(np.sqrt(np.mean(np.sum(
                        (proj.reshape(4, 2) - img_points) ** 2, axis=1))))
                    c = img_points[0]
                    cv2.putText(frame, f"id {marker_id}  z={t[2]:.3f} m  err={reproj:.2f} px",
                                (int(c[0]), int(c[1]) - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

            if self.publish_debug:
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        self.markers_pub.publish(out)
        if transforms:
            self.tf_broadcaster.sendTransform(transforms)

        if self.publish_debug:
            debug_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            debug_msg.header = msg.header
            self.debug_pub.publish(debug_msg)


def main(args=None):
    rclpy.init(args=args)
    node = ArucoDetector()
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