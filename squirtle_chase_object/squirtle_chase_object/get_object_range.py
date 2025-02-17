# Shahmeel Naseem, Evan Dodani

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan 
from std_msgs.msg._float32_multi_array import Float32MultiArray  # noqa: F401
from std_msgs.msg._float32 import Float32  # noqa: F401
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy

import numpy as np
import cv2
from cv_bridge import CvBridge

class MinimalVideoSubscriber(Node):
    def __init__(self):        
        # Creates the node.
        super().__init__('minimal_video_subscriber')
    
        # Set up QoS Profiles for passing images over WiFi
        image_qos_profile = QoSProfile(depth=10)
        image_qos_profile.history = QoSHistoryPolicy.KEEP_LAST
        image_qos_profile.durability = QoSDurabilityPolicy.VOLATILE 
        image_qos_profile.reliability = QoSReliabilityPolicy.BEST_EFFORT 

        # Declare that the minimal_video_subscriber node is subscribing to the /scan topic.
        self._LIDAR_subscriber = self.create_subscription(
            Float32MultiArray,
            '/scan',
            self._laser_callback,
            image_qos_profile)
        
        # Declare that the minimal_video_subscriber node is subscribing to the /coordinates topic.
        self._coord_subscriber = self.create_subscription(
            Float32MultiArray,
            '/coordinates',
            self._coord_callback,
            image_qos_profile)
        
        # Declare that the minimal_video_subscriber node is publishing to the /distance_error topic.
        self._dist_error_publisher = self.create_publisher(
            Float32,
            '/distance_error',
            10)
        
        self.bridge = CvBridge()
        self._coordinates = None
        self.timer = self.create_timer(0.1, self.timer_callback)  # Timer to publish coordinates periodically
    
    def _laser_callback(self, lidar_msg):
        # get lidar_data and compute weighted average of distance based on xL_angle and xR_angle from self.coord
    
    def _coord_callback(self, coord_msg):
        self.coord = coord_msg.data
        
    
    def timer_callback(self):
        if self._coordinates is not None:
            coord_msg = 
            coord_msg.data = 
            self._coordinate_publisher.publish()
            self.get_logger().info()

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