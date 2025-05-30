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
        super().__init__('object_finder')

        # Image transport QoS setup
        image_qos_profile = QoSProfile(depth=1)
        image_qos_profile.history = QoSHistoryPolicy.KEEP_LAST
        image_qos_profile.durability = QoSDurabilityPolicy.VOLATILE 
        image_qos_profile.reliability = QoSReliabilityPolicy.BEST_EFFORT 

        # Subscribe to image stream
        self._pi_subscriber = self.create_subscription(
            CompressedImage,
            '/image_raw/compressed',
            self._image_callback,
            image_qos_profile
        )

        # Publish calculated object coordinates
        self._coordinate_publisher = self.create_publisher(
            Float32MultiArray,
            '/coordinates',
            10
        )

        self.bridge = CvBridge()
        self._coordinates = None
        self.timer = self.create_timer(0.1, self.timer_callback)

    def _image_callback(self, msg):
        # Convert compressed image to OpenCV format
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        height, width, _ = frame.shape

        # HSV segmentation
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_hsv = (0, 142, 113)
        upper_hsv = (12, 222, 193)
        mask = cv2.inRange(frame_hsv, lower_hsv, upper_hsv)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        coord = None
        xL = xR = None
        min_contour_area = 5000

        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) >= min_contour_area:
                x, y, w, h = cv2.boundingRect(largest)
                xL = width // 2 - float(x)
                xR = width // 2 - float(x + w)
                centroid_x = x + w // 2
                centroid_y = y + h // 2
                xC = float(width // 2 - centroid_x)

                if w >= 1 and h >= 1:
                    # Convert pixel offsets to camera-relative angles
                    xL_angle = xL * (62.2 / width)
                    xR_angle = xR * (62.2 / width)
                    xC_angle = xC * (62.2 / width)

                    coord = [
                        float(centroid_x), float(centroid_y),
                        float(width // 2), float(height // 2),
                        float(xL_angle), float(xR_angle), float(xC_angle)
                    ]
                else:
                    coord = [float(width // 2), float(height // 2), float(width // 2), float(height // 2), 0.0, 0.0, 0.0]

        self._coordinates = coord

    def timer_callback(self):
        if self._coordinates is not None:
            coord_msg = Float32MultiArray()
            coord_msg.data = self._coordinates
            self._coordinate_publisher.publish(coord_msg)
            self.get_logger().info(
                'Publishing coords: %.2f, %.2f, %.2f, %.2f, %.2f, %.2f' % (
                    coord_msg.data[0], coord_msg.data[1], coord_msg.data[2],
                    coord_msg.data[3], coord_msg.data[4], coord_msg.data[5]
                )
            )


def main():
    rclpy.init()
    node = ObjectFinderNode()

    try:
        rclpy.spin(node)
    except SystemExit:
        rclpy.logging.get_logger("Object Finder Node").info("Shutting Down")

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
