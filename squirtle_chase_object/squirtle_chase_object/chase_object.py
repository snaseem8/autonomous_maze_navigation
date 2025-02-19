# Shahmeel Naseem, Evan Dodani

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float32MultiArray, Float32  # noqa: F401
from geometry_msgs.msg._twist import Twist  # noqa: F401
# from geometry_msgs.msg._point import Point  # noqa: F401
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy
import numpy as np
import cv2
from cv_bridge import CvBridge

class MinimalSubscriber(Node):
    def __init__(self):        
        # Creates the node.
        super().__init__('minimal_subscriber')

        # Declare some variables
        self.centroid = None
        self.angular_cmd_vel = None
        self.linear_cmd_velocity = None
        self.timer = self.create_timer(0.1, self.timer_callback) # 0.1 second timer
        
        # Set up QoS Profiles for passing data over WiFi
        image_qos_profile = QoSProfile(depth=10)
        image_qos_profile.history = QoSHistoryPolicy.KEEP_LAST
        image_qos_profile.durability = QoSDurabilityPolicy.VOLATILE 
        image_qos_profile.reliability = QoSReliabilityPolicy.BEST_EFFORT 

        # Declare that the centroid_subscriber node is subscribing to the /coordinates topic.
        self.centroid_subscriber = self.create_subscription(
            Float32MultiArray,
            '/coordinates',
            self.angular_vel_callback,
            image_qos_profile)
        self.centroid_subscriber  # Prevents unused variable warning.
        
        # Declare that the centroid_subscriber node is subscribing to the /object_distance topic.
        self.centroid_subscriber = self.create_subscription(
            Float32,
            '/object_distance',
            self.linear_vel_callback,
            image_qos_profile)
        self.centroid_subscriber  # Prevents unused variable warning.

        # Declare that the centroid_subscriber node is publishing to the /cmd_vel topic.
        self.cmd_vel_publisher = self.create_publisher(
            Twist,
            '/cmd_vel',
            10)
        self.cmd_vel_publisher  # Prevents unused variable warning.

    def angular_vel_callback(self, centroid_msg):    
        centroid = centroid_msg.data
        
        if centroid is not None:
            # calculate error between the object centroid and the center of the camera view
            center_camera_x = centroid[2]
            center_camera_y = centroid[3]
            error = centroid[0] - center_camera_x
            if abs(error) > 30:
                # calculate input to rotate the robot
                self.angular_cmd_vel = (-0.01) * error
            else:
                self.angular_cmd_vel = 0.0
        else:
            self.angular_cmd_vel = 0.0

        if self.angular_cmd_vel > 2.5:
            self.angular_cmd_vel = 2.5
            
    def linear_vel_callback(self, obj_dist):
        # add P controller for calculating linear_cmd_velocity
        obj_dist = obj_dist.data
        if obj_dist is not None:
            desired_distance = 0.6
            dist_error = obj_dist - desired_distance  # desired distance from the object (meters)
            linear_gain = 0.3
            self.linear_cmd_velocity = (linear_gain) * dist_error  # P controller
            if self.linear_cmd_velocity > 0.15:
                self.linear_cmd_velocity = 0.15
            elif self.linear_cmd_velocity < -0.15:
                self.linear_cmd_velocity = -0.15
        else:
            self.linear_cmd_velocity = 0.0
            

    def timer_callback(self):
        if (self.angular_cmd_vel is not None) and (self.linear_cmd_velocity is not None):
            twist_msg = Twist()
            twist_msg.linear.x = self.linear_cmd_velocity
            twist_msg.linear.y = 0.0
            twist_msg.linear.z = 0.0
            twist_msg.angular.x = 0.0
            twist_msg.angular.y = 0.0
            twist_msg.angular.z = self.angular_cmd_vel
            
            self.cmd_vel_publisher.publish(twist_msg)
            self.get_logger().info('Publishing linear velocity (x,y,z): (%.2f, %.2f, %.2f)\n Publishing angular velocity (x,y,z): (%.2f, %.2f, %.2f)' % (twist_msg.linear.x, twist_msg.linear.y, twist_msg.linear.z, twist_msg.angular.x, twist_msg.angular.y, twist_msg.angular.z))
            

def main():
    rclpy.init()  # Init routine needed for ROS2.
    video_subscriber = MinimalSubscriber()  # Create class object to be used.
    
    try:
        rclpy.spin(video_subscriber)  # Trigger callback processing.       
    except SystemExit:
        rclpy.logging.get_logger("Camera Viewer Node Info...").info("Shutting Down")
    
    # Clean up and shutdown.
    video_subscriber.destroy_node()  
    rclpy.shutdown()


if __name__ == '__main__':
    main()
