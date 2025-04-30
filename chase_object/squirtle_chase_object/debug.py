import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy

import cv2
import numpy as np
from cv_bridge import CvBridge


class VideoDebugSubscriber(Node):
    def __init__(self):
        super().__init__('video_debug_subscriber')

        # Configurable display parameters
        self.declare_parameter('show_image_bool', True)
        self.declare_parameter('window_name', "Raw Image")
        self._display_image = bool(self.get_parameter('show_image_bool').value)
        self._titleOriginal = self.get_parameter('window_name').value

        # Initialize OpenCV display window
        if self._display_image:
            cv2.namedWindow(self._titleOriginal, cv2.WINDOW_AUTOSIZE)
            cv2.moveWindow(self._titleOriginal, 50, 50)

        # Initial HSV filter range (modifiable via click event if enabled)
        self.lower_hsv = np.array([0, 142, 113])
        self.upper_hsv = np.array([12, 222, 193])

        # Image QoS profile for compressed video stream
        image_qos_profile = QoSProfile(depth=5)
        image_qos_profile.history = QoSHistoryPolicy.KEEP_LAST
        image_qos_profile.durability = QoSDurabilityPolicy.VOLATILE
        image_qos_profile.reliability = QoSReliabilityPolicy.BEST_EFFORT

        # Subscribe to camera topic
        self._video_subscriber = self.create_subscription(
            CompressedImage,
            '/image_raw/compressed',
            self._image_callback,
            image_qos_profile
        )

        self.frame = None  # Latest image frame

    def _image_callback(self, msg):
        self.frame = CvBridge().compressed_imgmsg_to_cv2(msg, "bgr8")
        height, width, _ = self.frame.shape

        # Convert to HSV and threshold
        frame_hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(frame_hsv, self.lower_hsv, self.upper_hsv)

        # Detect contours of masked regions
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w >= 50 or h >= 50:
                cv2.rectangle(self.frame, (x, y), (x + w, y + h), (0, 255, 20), 3)

        # Show frame
        if self._display_image:
            cv2.imshow(self._titleOriginal, self.frame)
            if cv2.waitKey(1) == ord('q'):
                cv2.destroyAllWindows()


def main():
    rclpy.init()
    node = VideoDebugSubscriber()

    try:
        rclpy.spin(node)
    except SystemExit:
        rclpy.logging.get_logger("Camera Viewer Node").info("Shutting Down")

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
