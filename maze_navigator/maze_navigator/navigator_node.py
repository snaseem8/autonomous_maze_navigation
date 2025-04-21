#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32, Float32, Float32MultiArray
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy
import numpy as np
import tf2_ros
import math
import time

class NavigatorNode(Node):
    def __init__(self):
        super().__init__('navigator_node')
        
        # QoS Profile for LIDAR
        image_qos_profile = QoSProfile(depth=10)
        image_qos_profile.history = QoSHistoryPolicy.KEEP_LAST
        image_qos_profile.durability = QoSDurabilityPolicy.VOLATILE
        image_qos_profile.reliability = QoSReliabilityPolicy.BEST_EFFORT

        # QoS Profile for AMCL - THIS IS CRITICAL
        amcl_qos_profile = QoSProfile(depth=5)
        amcl_qos_profile.reliability = QoSReliabilityPolicy.RELIABLE
        amcl_qos_profile.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL  # Important to get last published pose
        
        # Update AMCL subscription with correct QoS
        self.amcl_sub = self.create_subscription(
            PoseWithCovarianceStamped, 
            '/amcl_pose', 
            self.amcl_callback, 
            amcl_qos_profile)

        # Add a direct controller timer to ensure regular control updates
        self.controller_timer = self.create_timer(0.1, self.controller_timer_callback)

        # Parameters
        self.cone_angle = np.deg2rad(14.0)  # 14 deg cone for LIDAR
        self.target_distance = 0.5
        self.linear_speed_max = 0.15  # m/s max
        self.angular_speed_max = 1.5  # rad/s max
        self.kp_linear = 0.4  # P gain for linear error
        self.kp_angular = 1.0  # P gain for angular error
        self.kp_forward_angular = 0.45  # P gain for angular error when in forward state
        self.angular_tolerance = np.deg2rad(5.0)  # 5 deg
        self.class_votes = []
        self.vote_threshold = 2      # require ≥2 votes
        self.last_class     = None   # for fallback single vote
        self.initial_class_done = False   # checks if we are starting in idle
        self.stabilization_time = 1.5  # seconds to wait after turning
        self.turn_complete_time = None  # will store the timestamp when turn completes
        
        # Wall alignment parameters
        self.wall_alignment_fov = np.deg2rad(30.0)  # 30 degree field of view
        self.kp_wall_alignment = 1.0  # P gain for wall alignment
        self.wall_alignment_tolerance = np.deg2rad(3.0)  # 3 degrees tolerance
        self.post_alignment_time = 1.5  # seconds to wait after alignment
        self.alignment_complete_time = None  # timestamp when alignment completes


        # State: IDLE, TURNING, STABILIZING, MOVING_FORWARD, ALIGNING
        self.state = 'IDLE'  

        # Subscribers
        self.class_sub = self.create_subscription(
            Int32, '/sign_class', self.class_callback, 10)
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, image_qos_profile)
        self.amcl_sub = self.create_subscription(PoseWithCovarianceStamped, '/amcl_pose', self.amcl_callback, 10)
        self.centroid_subscriber = self.create_subscription(
            Float32MultiArray,
            '/coordinates',
            self.coordinate_callback,
            image_qos_profile)
        
        # Publisher
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.front_dist_pub = self.create_publisher(Float32, '/front_dist', 10)
        
        # State
        self.front_distance = float('inf')
        self.current_class = None
        self.goal_reached = False
        self.current_pose = (0.0, 0.0, 0.0)  # (x, y, yaw)
        self.target_yaw = None  # Target yaw for turning
        self.ignore_classification = False
        self.perform_startup_wiggle()  # starts the wiggle
        
    def perform_startup_wiggle(self):
        """Perform a simple left-right wiggle at startup to help AMCL initialize"""
        self.get_logger().info('*** STARTING WIGGLE NOW ***')
        
        # First rotate left
        cmd = Twist()
        cmd.angular.z = 0.5
        self.cmd_vel_pub.publish(cmd)
        
        # Keep publishing for a second to ensure it's received
        for _ in range(10):
            self.cmd_vel_pub.publish(cmd)
            time.sleep(0.1)
        
        # Now rotate right
        cmd.angular.z = -0.5
        self.cmd_vel_pub.publish(cmd)
        
        # Keep publishing for a second
        for _ in range(10):
            self.cmd_vel_pub.publish(cmd)
            time.sleep(0.1)
        
        # Stop
        cmd.angular.z = 0.0
        self.cmd_vel_pub.publish(cmd)
        
        self.get_logger().info('*** WIGGLE COMPLETE ***')

    def execute_wiggle_step(self, angular_vel):
        """Execute a single step of the wiggle sequence"""
        cmd = Twist()
        cmd.angular.z = angular_vel
        self.cmd_vel_pub.publish(cmd)
        
        # If this is the final step (stopped), continue with normal processing
        if angular_vel == 0.0:
            self.get_logger().info('Wiggle complete, ready for normal operation')

    def quaternion_to_yaw(self, quaternion):
        x, y, z, w = quaternion
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return yaw
        
    def controller_timer_callback(self):
        """Run controllers based on current state at a regular interval"""
        if self.state == 'TURNING' and self.target_yaw is not None:
            self.turn_controller()
        elif self.state == 'STABILIZING' and self.turn_complete_time is not None:
            # Check if stabilization time has elapsed
            current_time = self.get_clock().now().seconds_nanoseconds()[0]
            if current_time - self.turn_complete_time >= self.stabilization_time:
                # Transition to forward movement
                self.state = 'MOVING_FORWARD'
                self.get_logger().info('Camera stabilized, moving forward to next wall')
        elif self.state == 'MOVING_FORWARD' and self.target_yaw is not None:
            self.drive_controller()
        elif self.state == 'ALIGNING':
            self.wall_alignment_controller()
        elif self.state == 'ALIGNMENT_STABILIZING':
            # Check if post-alignment stabilization time has elapsed
            current_time = self.get_clock().now().seconds_nanoseconds()[0]
            if current_time - self.alignment_complete_time >= self.post_alignment_time:
                # Transition to sign processing
                self.get_logger().info('Post-alignment stabilization complete, processing sign')
                # Simply transition to IDLE and call the existing function to process votes
                self.state = 'IDLE'
                self.target_yaw = None
                self.ignore_classification = False
                
                # Use your existing code for handling votes and process_sign()
                # (You already have this code in your existing implementation)
                n_votes = len(self.class_votes)
                if n_votes >= self.vote_threshold:
                    most_common = max(set(self.class_votes), key=self.class_votes.count)
                    self.current_class = most_common
                else:
                    self.current_class = self.last_class
                
                self.class_votes.clear()
                self.process_sign()

    def amcl_callback(self, msg: PoseWithCovarianceStamped):
        try:
            # extract x, y
            x = msg.pose.pose.position.x
            y = msg.pose.pose.position.y
            # convert quaternion → yaw
            # Convert quaternion to yaw
            quat = msg.pose.pose.orientation
            yaw = self.quaternion_to_yaw([quat.x, quat.y, quat.z, quat.w])
            
            # Log detailed AMCL data
            if x != 0.0 or y != 0.0 or yaw != 0.0:
                self.get_logger().info(f"AMCL updated: pos=({x:.3f}, {y:.3f}), yaw={yaw:.3f}")
            
            self.current_pose = (x, y, yaw)

            # Check if we have a valid target_yaw before logging
            if self.target_yaw is not None:
                self.get_logger().info(f'target yaw: {self.target_yaw:.2f}')
            
            # run whichever controller is active
            if self.state == 'TURNING' and self.target_yaw is not None:
                self.turn_controller()
            elif self.state == 'MOVING_FORWARD' and self.target_yaw is not None:
                self.drive_controller()
        except Exception as e:
            self.get_logger().error(f'Error in AMCL callback: {e}')
            
    def normalize_angle(self, angle):
        return math.atan2(math.sin(angle), math.cos(angle))

    def coordinate_callback(self, msg):
        self.coord = msg.data
        
    def scan_callback(self, msg):
        # Compute front distance (14° cone)
        forward_angle = 0.0
        cone_half_width = self.cone_angle / 2
        angles = np.array([msg.angle_min + i * msg.angle_increment for i in range(len(msg.ranges))])
        cone_indices = np.where((angles >= forward_angle - cone_half_width) &
                                (angles <= forward_angle + cone_half_width))[0]
        valid_distances = [msg.ranges[i] for i in cone_indices if np.isfinite(msg.ranges[i]) and
                           msg.range_min <= msg.ranges[i] <= msg.range_max]
        self.front_distance = np.mean(valid_distances)
        
        dist_msg = Float32()
        dist_msg.data = self.front_distance
        self.front_dist_pub.publish(dist_msg)
        
        # One‐time initial classification if we start in IDLE and are at a wall
        if (not self.initial_class_done
            and self.state == 'IDLE'
            and self.front_distance <= self.target_distance):
            
            if self.last_class is None:
                self.get_logger().warn('Initial wall detected but no classifier msg yet.')
            else:
                self.get_logger().info(f'Initial classification from last_class → {self.last_class}')
                self.current_class = self.last_class
                self.initial_class_done = True
                # take your normal action
                self.process_sign()
            # skip the rest of scan_callback on this cycle
            return

        # Check if we are close enough to the wall to stop
        if self.front_distance <= self.target_distance and self.state == 'MOVING_FORWARD':
            # stop and reset state
            self.stop_robot()
            
            # Enter ALIGNING state instead of IDLE 
            self.state = 'ALIGNING'
            self.get_logger().info('Reached wall, aligning perpendicular to wall...')
            
            # Store wall scan data for alignment
            self.wall_scan = msg
            
            # Skip the rest of the scan_callback
            return
        
        # Update wall scan data if in ALIGNING state
        if self.state == 'ALIGNING':
            self.wall_scan = msg

    def wall_alignment_controller(self):
        """Align robot perpendicular to the wall"""
        if not hasattr(self, 'wall_scan'):
            self.get_logger().warn('No wall scan data available for alignment')
            self.state = 'IDLE'  # Fall back to IDLE state
            return
        
        # Find closest point in field of view
        angles = np.array([self.wall_scan.angle_min + i * self.wall_scan.angle_increment 
                          for i in range(len(self.wall_scan.ranges))])
        fov_half = self.wall_alignment_fov / 2
        wall_indices = np.where((angles >= -fov_half) & (angles <= fov_half))[0]
        
        # Find the closest valid point and its angle
        min_distance = float('inf')
        min_angle = 0.0
        
        for i in wall_indices:
            if (np.isfinite(self.wall_scan.ranges[i]) and 
                self.wall_scan.range_min <= self.wall_scan.ranges[i] <= self.wall_scan.range_max and
                self.wall_scan.ranges[i] < min_distance):
                min_distance = self.wall_scan.ranges[i]
                min_angle = angles[i]
        
        # If we found a valid closest point
        if min_distance < float('inf'):
            # To be perpendicular to wall, closest point should be at angle 0
            # If min_angle is positive, turn right (negative angular velocity)
            # If min_angle is negative, turn left (positive angular velocity)
            
            self.get_logger().info(f'Wall alignment: closest point at angle {min_angle:.4f}')
            
            # When alignment is complete:
            if abs(min_angle) <= self.wall_alignment_tolerance:
                self.get_logger().info('Wall alignment complete, starting post-alignment stabilization')
                self.stop_robot()
                
                # Record the time and transition to stabilizing
                self.alignment_complete_time = self.get_clock().now().seconds_nanoseconds()[0]
                self.state = 'ALIGNMENT_STABILIZING'
            else:
                # Apply P controller for alignment
                angular_vel = -self.kp_wall_alignment * min_angle
                angular_vel = max(min(angular_vel, self.angular_speed_max/2), -self.angular_speed_max/2)
                
                cmd = Twist()
                cmd.angular.z = angular_vel
                self.cmd_vel_pub.publish(cmd)
                
                self.get_logger().info(f'Aligning to wall: angle_err={min_angle:.3f}, cmd={angular_vel:.3f}')
        else:
            # If no valid points found, skip alignment
            self.get_logger().warn('No valid wall points found for alignment')
            self.state = 'IDLE'
            self.process_sign()
        
    def class_callback(self, msg):
        # Always store last_class, regardless of state
        self.last_class = msg.data
        
        # Collect votes during approach
        if self.state == 'MOVING_FORWARD' and 0.4 < self.front_distance <= 1.0:
            self.class_votes.append(msg.data)
            self.get_logger().info(f'Collected class vote: {msg.data}')
        
        # Only process sign class in IDLE state when not ignoring
        if self.goal_reached or self.state != 'IDLE' or self.ignore_classification:
            return
            
        self.current_class = msg.data
        self.get_logger().info(f'Received sign class: {self.current_class}')
        self.process_sign()

            
    def process_sign(self):
        self.get_logger().info(f'current_pose: {self.current_pose}')
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
            self.target_yaw = self.normalize_angle(self.target_yaw)
        
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
                # Enter stabilization state instead of directly to MOVING_FORWARD
                self.state = 'STABILIZING'
                self.turn_complete_time = self.get_clock().now().seconds_nanoseconds()[0]
                self.get_logger().info('Turn complete, stabilizing camera...')
            else:
                self.state = 'IDLE'
                self.target_yaw = None
                self.ignore_classification = False
                self.get_logger().info('Goal reached, navigation stopped')

    def transition_to_forward(self):
        """Transition to MOVING_FORWARD state after camera stabilization"""
        self.state = 'MOVING_FORWARD'
        self.get_logger().info('Camera stabilized, moving forward to next wall')
            
    def drive_controller(self):
        if self.current_pose is None or self.target_yaw is None:
            self.get_logger().warn('Drive controller called with missing pose or target_yaw')
            return
            
        if self.front_distance > self.target_distance:
            # forward P‐control
            linear_vel = min(self.kp_linear * (self.front_distance - (self.target_distance - 0.1)),
                            self.linear_speed_max)

            # small angular correction to hold heading
            _, _, curr_yaw = self.current_pose
            yaw_error = self.normalize_angle(self.target_yaw - curr_yaw)
            angular_vel = self.kp_forward_angular * yaw_error
            angular_vel = max(min(angular_vel, self.angular_speed_max),
                            -self.angular_speed_max)

            cmd = Twist()
            cmd.linear.x = linear_vel
            cmd.angular.z = angular_vel
            self.cmd_vel_pub.publish(cmd)
            self.get_logger().info(f'Driving → dist {self.front_distance:.2f}m, yaw_err {yaw_error:.2f}rad')
        
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