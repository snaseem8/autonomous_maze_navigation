import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float32MultiArray, Float32
from geometry_msgs.msg import Twist
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy
import numpy as np
import cv2
from cv_bridge import CvBridge


class ObjectTrackerNode(Node):
    def __init__(self):        
        super().__init__('object_tracker')

        # Internal state variables
        self.centroid = None
        self.angular_cmd_vel = None
        self.linear_cmd_velocity = None
        self.linear_error = 0.0
        self.desired_distance = 0.5  # meters
        self.obj_dist = 0

        # Timer callback for sending velocity commands
        self.timer = self.create_timer(0.1, self.timer_callback)

        # QoS settings for reliable image/data transport
        image_qos_profile = QoSProfile(depth=10)
        image_qos_profile.history = QoSHistoryPolicy.KEEP_LAST
        image_qos_profile.durability = QoSDurabilityPolicy.VOLATILE
        image_qos_profile.reliability = QoSReliabilityPolicy.BEST_EFFORT

        # Subscribe to object centroid topic
        self.create_subscription(
            Float32MultiArray,
            '/coordinates',
            self.angular_vel_callback,
            image_qos_profile
        )

        # Subscribe to object distance topic
        self.create_subscription(
            Float32,
            '/object_distance',
            self.linear_vel_callback,
            image_qos_profile
        )

        # Publisher to control robot velocity
        self.cmd_vel_publisher = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )

    def angular_vel_callback(self, centroid_msg):    
        centroid = centroid_msg.data

        if centroid is not None:
            center_camera_x = centroid[2]
            error = centroid[0] - center_camera_x

            # Adjust angular velocity based on distance error and threshold
            if self.linear_error < 0:
                if abs(error) > np.floor(25 / self.obj_dist):
                    self.angular_cmd_vel = -0.0075 * error
                else:
                    self.angular_cmd_vel = 0.0
            else:
                if abs(error) > 30:
                    self.angular_cmd_vel = -0.0075 * error
                else:
                    self.angular_cmd_vel = 0.0
        else:
            self.angular_cmd_vel = 0.0

        # Clamp angular velocity
        self.angular_cmd_vel = max(min(self.angular_cmd_vel, 2), -2)

    def linear_vel_callback(self, obj_dist):
        obj_dist = obj_dist.data
        self.obj_dist = obj_dist

        if obj_dist is not None:
            dist_error = obj_dist - self.desired_distance
            self.linear_error = dist_error
            # Gain tuning for forward/backward motion
            linear_gain = 0.15 if dist_error > 0 else 0.35
            self.linear_cmd_velocity = linear_gain * dist_error
            # Clamp linear velocity
            self.linear_cmd_velocity = max(min(self.linear_cmd_velocity, 0.15), -0.15)
        else:
            self.linear_cmd_velocity = 0.0

    def timer_callback(self):
        # Publish Twist message if velocities are ready
        if self.angular_cmd_vel is not None and self.linear_cmd_velocity is not None:
            twist_msg = Twist()
            twist_msg.linear.x = self.linear_cmd_velocity
            twist_msg.angular.z = self.angular_cmd_vel

            self.cmd_vel_publisher.publish(twist_msg)
            self.get_logger().info(
                'Publishing linear x: %.2f, angular z: %.2f' % (
                    twist_msg.linear.x, twist_msg.angular.z
                )
            )

def main():
    rclpy.init()
    node = ObjectTrackerNode()

    try:
        rclpy.spin(node)
    except SystemExit:
        rclpy.logging.get_logger("Object Tracker").info("Shutting Down")

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
