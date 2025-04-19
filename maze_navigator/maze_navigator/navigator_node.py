#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy
import numpy as np
import tf2_ros
import math

class NavigatorNode(Node):
    def __init__(self):
        super().__init__('navigator_node')
        
        # QoS Profile for LIDAR
        image_qos_profile = QoSProfile(depth=10)
        image_qos_profile.history = QoSHistoryPolicy.KEEP_LAST
        image_qos_profile.durability = QoSDurabilityPolicy.VOLATILE
        image_qos_profile.reliability = QoSReliabilityPolicy.BEST_EFFORT
        
        # Parameters
        self.cone_angle = np.deg2rad(14.0)  # 14 deg cone for LIDAR
        self.target_distance = 0.5
        self.linear_speed_max = 0.15  # m/s max
        self.angular_speed_max = 1.5  # rad/s max
        self.kp_linear = 0.5  # P gain for linear error
        self.kp_angular = 1.0  # P gain for angular error
        self.angular_tolerance = np.deg2rad(5.0)  # 5 deg
        
        # Subscribers
        self.class_sub = self.create_subscription(
            Int32, '/sign_class', self.class_callback, 10)
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, image_qos_profile)
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10)
        
        # Publisher
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # State
        self.front_distance = float('inf')
        self.current_class = None
        self.state = 'IDLE'  # IDLE, TURNING, MOVING_FORWARD
        self.goal_reached = False
        self.current_pose = None  # (x, y, yaw)
        self.target_yaw = None  # Target yaw for turning
        self.ignore_classification = False
        
    def quaternion_to_yaw(self, quaternion):
        x, y, z, w = quaternion
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return yaw
        
    def odom_callback(self, msg):
        # Extract pose (x, y, yaw) from /odom
        x_current = msg.pose.pose.position.x
        y_current = msg.pose.pose.position.y
        orientation = msg.pose.pose.orientation
        quaternion = [orientation.x, orientation.y, orientation.z, orientation.w]
        yaw = self.quaternion_to_yaw(quaternion)
        self.current_pose = (x_current, y_current, yaw)
        
        # Run controllers based on state
        if self.state == 'TURNING' and self.target_yaw is not None:
            self.turn_controller()
        elif self.state == 'MOVING_FORWARD':
            self.drive_controller()
        
    def scan_callback(self, msg):
        # Compute front distance (14° cone)
        forward_angle = 0.0
        cone_half_width = self.cone_angle / 2
        angles = np.array([msg.angle_min + i * msg.angle_increment for i in range(len(msg.ranges))])
        cone_indices = np.where((angles >= forward_angle - cone_half_width) &
                                (angles <= forward_angle + cone_half_width))[0]
        valid_distances = [msg.ranges[i] for i in cone_indices if np.isfinite(msg.ranges[i]) and
                           msg.range_min <= msg.ranges[i] <= msg.range_max]
        self.front_distance = np.mean(valid_distances) if valid_distances else 10.0
        
        # Stop if too close to wall
        if self.front_distance <= self.target_distance and self.state == 'MOVING_FORWARD':
            self.stop_robot()
            self.state = 'IDLE'
            self.target_yaw = None
            self.ignore_classification = False
            self.get_logger().info('Reached wall, ready for next classification')
        
    def class_callback(self, msg):
        if self.goal_reached or self.state != 'IDLE' or self.ignore_classification:
            return
        self.current_class = msg.data
        self.get_logger().info(f'Received sign class: {self.current_class}')
        self.process_sign()
            
    def process_sign(self):
        # self.get_logger().info(f'class: {self.current_class}')
        # self.get_logger().info(f'goal_reached: {self.goal_reached}')
        # self.get_logger().info(f'current_pose: {self.current_pose}')
        if self.current_class is None or self.goal_reached or self.current_pose is None:
            return
        
        self.state = 'TURNING'
        self.ignore_classification = True
        _, _, current_yaw = self.current_pose
        
        # Sign classes: 0=empty (recovery), 1=left, 2=right, 3=do not enter, 4=stop, 5=goal
        if self.current_class == 5:  # Goal
            self.get_logger().info('Goal reached!')
            self.goal_reached = True
            self.stop_robot()
            self.state = 'IDLE'
            self.target_yaw = None
            self.ignore_classification = False
        elif self.current_class == 0:  # Empty wall (recovery)
            # Turn 90° left
            self.target_yaw = current_yaw + np.pi / 2
            self.get_logger().info(f'Recovery: turning left to yaw {self.target_yaw:.2f}')
        elif self.current_class == 1:  # Left
            # Turn 90° left
            self.target_yaw = current_yaw + np.pi / 2
            self.get_logger().info(f'Turning left to yaw {self.target_yaw:.2f}')
        elif self.current_class == 2:  # Right
            # Turn 90° right
            self.target_yaw = current_yaw - np.pi / 2
            self.get_logger().info(f'Turning right to yaw {self.target_yaw:.2f}')
        elif self.current_class in [3, 4]:  # Do not enter or Stop
            # Turn 180°
            self.target_yaw = current_yaw + np.pi
            self.get_logger().info(f'Turning around to yaw {self.target_yaw:.2f}')
        else:
            self.get_logger().error(f'Unknown sign class: {self.current_class}')
            self.state = 'IDLE'
            self.target_yaw = None
            self.ignore_classification = False
            return
        
        # Normalize target yaw
        if self.target_yaw is not None:
            self.target_yaw = np.arctan2(np.sin(self.target_yaw), np.cos(self.target_yaw))
        
    def turn_controller(self):
        if self.current_pose is None or self.target_yaw is None:
            return
        
        # Current yaw
        _, _, curr_yaw = self.current_pose
        
        # Angular error (shortest path)
        angular_error = np.arctan2(np.sin(self.target_yaw - curr_yaw), np.cos(self.target_yaw - curr_yaw))
        
        # P controller
        angular_vel = self.kp_angular * angular_error
        angular_vel = max(min(angular_vel, self.angular_speed_max), -self.angular_speed_max)
        
        # Publish command
        cmd = Twist()
        cmd.angular.z = angular_vel if abs(angular_error) > self.angular_tolerance else 0.0
        self.cmd_vel_pub.publish(cmd)
        
        # Check if turn complete
        if abs(angular_error) <= self.angular_tolerance:
            self.stop_robot()
            if not self.goal_reached:
                self.state = 'MOVING_FORWARD'
                self.get_logger().info('Turn complete, moving forward to next wall')
            else:
                self.state = 'IDLE'
                self.target_yaw = None
                self.ignore_classification = False
                self.get_logger().info('Goal reached, navigation stopped')
        
    def drive_controller(self):
        if self.front_distance > self.target_distance:
            # Drive forward
            linear_vel = min(self.kp_linear * (self.front_distance - self.target_distance), self.linear_speed_max)
            cmd = Twist()
            cmd.linear.x = linear_vel
            self.cmd_vel_pub.publish(cmd)
            self.get_logger().info(f'Driving forward, distance to wall: {self.front_distance:.2f}m')
        else:
            # Stop (redundant, handled by scan_callback)
            self.stop_robot()
            self.state = 'IDLE'
            self.target_yaw = None
            self.ignore_classification = False
            self.get_logger().info('Reached wall, ready for next classification')
        
    def stop_robot(self):
        cmd = Twist()
        self.cmd_vel_pub.publish(cmd)
        
def main(args=None):
    rclpy.init(args=args)
    node = NavigatorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()