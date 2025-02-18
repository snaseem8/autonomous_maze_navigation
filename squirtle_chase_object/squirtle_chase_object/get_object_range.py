# Shahmeel Naseem, Evan Dodani

import rclpy
from rclpy.node import Node
from sensor_msgs.msg._laser_scan import LaserScan  # noqa: F401
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
        
        # Declare that the minimal_video_subscriber node is publishing to the /object_distance topic.
        self._dist_publisher = self.create_publisher(
            Float32,
            '/object_distance',
            10)
        
        self.bridge = CvBridge()
        self._coordinates = None
        self.object_dist = None
        self.timer = self.create_timer(0.1, self.timer_callback)  # Timer to publish coordinates periodically
    
    def _coord_callback(self, coord_msg):
        self._coordinates = coord_msg.data  # pull in the coordinates from the /coordinates topic

    def _laser_callback(self, lidar_msg):
        # get lidar_data and compute weighted average of distance based on xL_angle and xR_angle from self.coord
        desired_dist = 0.6  # desired distance from the object (meters)

        ranges = np.array(lidar_msg.ranges)   # distance data from the lidar scan [m], with it indexed. Corresponds to an angle value
        ranges[(ranges < 0.1) | (ranges > 5)] = desired_dist  # mask to filter out the invalid data (like inf and NaN) also cap the max reliable distance to 10m
        # instead of changing invalid data to 0 or deleting them, we simply set them to the desired distance so that the robot wont move (no error)
        

        min_angle = lidar_msg.angle_min   # start angle of the lidar scan [rad]
        max_angle = lidar_msg.angle_max   # end angle of the lidar scan [rad]
        angle_increment = lidar_msg.angle_increment   # angular distance between measurements [rad]
        xL_angle = np.deg2rad(self._coordinates[4])  # left-most angle of the bounding box in the camera view [rad]
        xR_angle = np.deg2rad(self._coordinates[5]) # right-most angle of the bounding box in the camera view [rad]
        left_index = int(np.ceil((xL_angle - min_angle)/angle_increment))  # calculates the index of the left angle, rounds up to integer - this is to be safe
        right_index = int(np.floor((xR_angle - min_angle)/angle_increment))   # calculates the index of the right angle, rounds down to integer
        avg_dist = np.mean(ranges[left_index:right_index])  # calculates the average of the distances between the left and right angles
        self.object_dist = avg_dist

    
    def timer_callback(self):
        if self._coordinates & self.object_dist is not None:
            object_dist = Float32()
            object_dist.data = self.object_dist
            self._dist_publisher.publish(object_dist)
            self.get_logger().info('Publishing object distance (depth): %.2f)' % (object_dist.data))

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