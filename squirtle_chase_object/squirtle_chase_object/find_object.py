# Shahmeel Naseem, Evan Dodani

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg._float32_multi_array import Float32MultiArray  # noqa: F401
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy

import numpy as np
import cv2
from cv_bridge import CvBridge

class MinimalVideoSubscriber(Node):
    def __init__(self):        
        # Creates the node.
        super().__init__('minimal_video_subscriber')
    
        # Set up QoS Profiles for passing images over WiFi
        image_qos_profile = QoSProfile(depth=1)
        image_qos_profile.history = QoSHistoryPolicy.KEEP_LAST
        image_qos_profile.durability = QoSDurabilityPolicy.VOLATILE 
        image_qos_profile.reliability = QoSReliabilityPolicy.BEST_EFFORT 

        # Declare that the minimal_video_subscriber node is subscribing to the /camera/image/compressed topic.
        self._pi_subscriber = self.create_subscription(
            CompressedImage,
            '/image_raw/compressed',
            self._image_callback,
            image_qos_profile)
        
        # Declare that the minimal_video_subscriber node is publishing to the /coordinates topic.
        self._coordinate_publisher = self.create_publisher(
            Float32MultiArray,
            '/coordinates',
            10)
        
        self.bridge = CvBridge()
        self._coordinates = None
        self.timer = self.create_timer(0.1, self.timer_callback)  # Timer to publish coordinates periodically
    
    def _image_callback(self, pi_image):
        # Convert ROS2 CompressedImage message to OpenCV image
        np_arr = np.frombuffer(pi_image.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        height, width, _ = frame.shape

        # Convert to HSV
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_hsv = (0, 142, 113)
        upper_hsv = (12, 222, 193)

        # Create mask and find contours
        mask = cv2.inRange(frame_hsv, lower_hsv, upper_hsv)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        coord = None  # Default value in case no contours are found
        xL = None
        xR = None
        min_contour_area = 5000  # Minimum area for a contour to be considered
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)  # Find the biggest contour
            
            if cv2.contourArea(largest_contour) >= min_contour_area:  # Check if it's big enough
                x, y, w, h = cv2.boundingRect(largest_contour)
                xL = width // 2 - float(x)
                xR = width // 2 - float(x + w)
                centroid_x = x + w // 2
                centroid_y = y + h // 2
                xC = float(width//2 - centroid_x)
                if w >= 1 and h >= 1:
                    # Convert x_object pixel value to degrees
                    xL_angle =  float(xL * (62.2 / width)) # 62.2 degrees is the horizontal field of view of the camera
                    xR_angle = float(xR * (62.2 / width))
                    xC_angle = float(xC * (62.2 / width))
                    coord = [float(centroid_x), float(centroid_y), float(width//2), float(height//2), float(xL_angle), float(xR_angle), float(xC_angle)] #centroid x, centroid y, image center x, image center y
                else:
                    coord = [float(width//2), float(height//2), float(width//2), float(height//2), 0.0, 0.0, 0.0] # if bounding box returns 0s, set centroid to image center so there is no error
            
        self._coordinates = coord  # Store detected coordinates
    
    def timer_callback(self):
        if self._coordinates is not None:
            coord_msg = Float32MultiArray()
            coord_msg.data = self._coordinates
            self._coordinate_publisher.publish(coord_msg)
            self.get_logger().info('Publishing coordinates, image dimensions, and horiz angles: %.2f, %.2f, %.2f, %.2f, %.2f, %.2f)' % (coord_msg.data[0], coord_msg.data[1], coord_msg.data[2], coord_msg.data[3], coord_msg.data[4], coord_msg.data[5]))

def main():
    rclpy.init()  # Init routine needed for ROS2.
    video_subscriber = MinimalVideoSubscriber()  # Create class object to be used.
    
    try:
        rclpy.spin(video_subscriber)  # Trigger callback processing.        
    except SystemExit:
        rclpy.logging.get_logger("Camera Viewer Node Info...").info("Shutting Down")
    
    # Clean up and shutdown.
    video_subscriber.destroy_node()  
    rclpy.shutdown()


if __name__ == '__main__':
    main()