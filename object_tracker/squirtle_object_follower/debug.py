import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy
import cv2
import numpy as np
from cv_bridge import CvBridge

class ImageDebugNode(Node):
    def __init__(self):
        super().__init__('image_debug_node')

        # Declare display parameters
        self.declare_parameter('show_image_bool', True)
        self.declare_parameter('window_name', 'Raw Image')
        self._display_image = self.get_parameter('show_image_bool').value
        self._window_title = self.get_parameter('window_name').value

        # Initialize OpenCV window if display is enabled
        if self._display_image:
            cv2.namedWindow(self._window_title, cv2.WINDOW_AUTOSIZE)
            cv2.moveWindow(self._window_title, 50, 50)
            cv2.setMouseCallback(self._window_title, self._click_event)

        # Default HSV range
        self._lower_hsv = np.array([90, 50, 50])
        self._upper_hsv = np.array([130, 255, 255])

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

        self._bridge = CvBridge()
        self._current_frame = None

    def _click_event(self, event, x, y, flags, param):
        # Update HSV range on left-click
        if event == cv2.EVENT_LBUTTONDOWN and self._current_frame is not None:
            hsv_frame = cv2.cvtColor(self._current_frame, cv2.COLOR_BGR2HSV)
            h, s, v = hsv_frame[y, x]
            self._lower_hsv = np.array([max(0, h - 10), max(0, s - 40), max(0, v - 40)])
            self._upper_hsv = np.array([min(179, h + 10), min(255, s + 40), min(255, v + 40)])
            self.get_logger().info(f'Updated HSV range: lower={self._lower_hsv}, upper={self._upper_hsv}')

    def _image_callback(self, msg):
        # Process incoming image
        self._current_frame = self._bridge.compressed_imgmsg_to_cv2(msg, 'bgr8')

        if self._display_image:
            # Apply HSV thresholding
            hsv_frame = cv2.cvtColor(self._current_frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv_frame, self._lower_hsv, self._upper_hsv)

            # Find and draw bounding boxes for significant contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if w >= 50 or h >= 50:
                    cv2.rectangle(self._current_frame, (x, y), (x + w, y + h), (0, 255, 20), 3)

            # Display the processed frame
            cv2.imshow(self._window_title, self._current_frame)
            if cv2.waitKey(1) == ord('q'):
                cv2.destroyAllWindows()

def main():
    rclpy.init()
    node = ImageDebugNode()
    try:
        rclpy.spin(node)
    except SystemExit:
        node.get_logger().info('Shutting down')
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
