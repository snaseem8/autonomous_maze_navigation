import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float32MultiArray
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy
import numpy as np
import cv2
from cv_bridge import CvBridge

class ObjectFinderNode(Node):
    def __init__(self):
        super().__init__('object_finder_node')

        # Configure QoS for image subscription
        qos_profile = QoSProfile(
            depth=5,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            reliability=QoSReliabilityPolicy.BEST_EFFORT
        )

        # Subscribe to compressed image topic
        self._image_subscriber = self.create_subscription(
            CompressedImage,
            '/image_raw/compressed',
            self._image_callback,
            qos_profile
        )

        # Publish object coordinates
        self._coordinate_publisher = self.create_publisher(
            Float32MultiArray,
            '/coordinates',
            qos_profile
        )

        self._bridge = CvBridge()
        self._coordinates = None
        self._timer = self.create_timer(0.1, self._publish_coordinates)

    def _image_callback(self, msg):
        # Process image to find object centroid
        frame = self._bridge.compressed_imgmsg_to_cv2(msg, 'bgr8')
        height, width, _ = frame.shape

        # Apply HSV thresholding
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_hsv = np.array([0, 142, 113])
        upper_hsv = np.array([12, 222, 193])
        mask = cv2.inRange(frame_hsv, lower_hsv, upper_hsv)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        default_coord = [float(width // 2), float(height // 2), float(width // 2), float(height // 2)]

        # Process largest contour if significant
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            if w >= 1 and h >= 1 and cv2.contourArea(largest_contour) > 100:
                self._coordinates = [float(x + w // 2), float(y + h // 2), float(width // 2), float(height // 2)]
            else:
                self._coordinates = default_coord
        else:
            self._coordinates = default_coord

    def _publish_coordinates(self):
        # Publish object coordinates
        if self._coordinates is not None:
            msg = Float32MultiArray()
            msg.data = self._coordinates
            self._coordinate_publisher.publish(msg)
            self.get_logger().info(f'Coordinates: {msg.data[0]:.2f}, {msg.data[1]:.2f}, center: {msg.data[2]:.2f}, {msg.data[3]:.2f}')

def main():
    rclpy.init()
    node = ObjectFinderNode()
    try:
        rclpy.spin(node)
    except SystemExit:
        node.get_logger().info('Shutting down')
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
