# Shahmeel Naseem, Evan Dodani

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy

import cv2
import numpy as np
from cv_bridge import CvBridge

class MinimalVideoSubscriber(Node):
    def __init__(self):
        super().__init__('minimal_video_subscriber')

        # Set Parameters
        self.declare_parameter('show_image_bool', True)
        self.declare_parameter('window_name', "Raw Image")

        # Determine Window Showing Based on Input
        self._display_image = bool(self.get_parameter('show_image_bool').value)

        # Declare some variables
        self._titleOriginal = self.get_parameter('window_name').value
        if self._display_image:
            cv2.namedWindow(self._titleOriginal, cv2.WINDOW_AUTOSIZE)
            cv2.moveWindow(self._titleOriginal, 50, 50)
            cv2.setMouseCallback(self._titleOriginal, self._click_event)  # Set mouse click event

        # Default HSV range (placeholder, updated on click)
        self.lower_hsv = np.array([90, 50, 50])
        self.upper_hsv = np.array([130, 255, 255])

        # QoS for image streaming over WiFi
        image_qos_profile = QoSProfile(depth=5)
        image_qos_profile.history = QoSHistoryPolicy.KEEP_LAST
        image_qos_profile.durability = QoSDurabilityPolicy.VOLATILE
        image_qos_profile.reliability = QoSReliabilityPolicy.BEST_EFFORT

        # Subscribe to the /image_raw/compressed topic
        self._video_subscriber = self.create_subscription(
            CompressedImage,
            '/image_raw/compressed',
            self._image_callback,
            image_qos_profile
        )

        self.frame = None  # Store the latest frame

    def _click_event(self, event, x, y, flags, param):
        """ Callback function to get HSV value from a clicked pixel. """
        if event == cv2.EVENT_LBUTTONDOWN and self.frame is not None:
            hsv_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
            clicked_hsv = hsv_frame[y, x]  # Get HSV value at click
            h, s, v = clicked_hsv

            # Define a small range around the clicked HSV value
            self.lower_hsv = np.array([max(0, h - 10), max(0, s - 40), max(0, v - 40)])
            self.upper_hsv = np.array([min(179, h + 10), min(255, s + 40), min(255, v + 40)])

            print(f"Updated HSV Range: Lower {self.lower_hsv}, Upper {self.upper_hsv}")

    def _image_callback(self, CompressedImage):
        """ Callback function to process incoming images. """
        self.frame = CvBridge().compressed_imgmsg_to_cv2(CompressedImage, "bgr8")
        height, width, _ = self.frame.shape

        # Convert to HSV and apply thresholding with dynamic HSV range
        frame_hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(frame_hsv, self.lower_hsv, self.upper_hsv)

        # Find contours of detected object
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            if w >= 50 or h >= 50:
                cv2.rectangle(self.frame, (x, y), (x + w, y + h), (0, 255, 20), 3)

        # Display the processed frame
        cv2.imshow(self._titleOriginal, self.frame)

        if cv2.waitKey(1) == ord('q'):
            cv2.destroyAllWindows()

def main():
    rclpy.init()
    video_subscriber = MinimalVideoSubscriber()
    
    try:
        rclpy.spin(video_subscriber)
    except SystemExit:
        rclpy.logging.get_logger("Camera Viewer Node Info...").info("Shutting Down")
    
    video_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
