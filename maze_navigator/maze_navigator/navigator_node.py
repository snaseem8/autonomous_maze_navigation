#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from std_msgs.msg import Int32, Bool
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseStamped, Twist
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import Odometry
import numpy as np
import tf_transformations
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

class NavigatorNode(Node):
    def __init__(self):
        super().__init__('navigator_node')
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        
        # Subscribers
        qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST
        )
        self.sign_sub = self.create_subscription(
            Int32, '/sign_class', self.sign_callback, qos)
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, qos)
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, qos)
        
        # Publishers
        self.goal_pub = self.create_publisher(Bool, '/goal_reached', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # State machine
        self.state = 'IDLE'
        self.current_sign = None
        self.front_distance = float('inf')
        self.current_pose = None
        self.recovery_count = 0
        self.max_recovery_attempts = 3
        self.visited_cells = set()  # Track visited cells to avoid loops
        self.cone_angle = 10 * np.pi / 180  # ±10° cone for front distance
        self.classification_distance = 0.7  # Start classifying at 0.7m
        self.min_distance = 0.4  # Stop at 0.4m
        self.forward_speed = 0.1  # m/s for moving during classification
        
        # Parameters
        self.turn_angles = {
            1: np.pi / 2,   # Left: 90 deg
            2: -np.pi / 2,  # Right: -90 deg
            3: np.pi,       # Do not enter: 180 deg
            4: np.pi,       # Stop: 180 deg
            5: 0.0          # Goal: No turn
        }
        
    def odom_callback(self, msg):
        self.current_pose = msg.pose.pose
        
    def sign_callback(self, msg):
        self.current_sign = msg.data
        if self.state == 'CLASSIFY':
            self.process_sign()
            
    def scan_callback(self, msg):
        # Compute indices for ±10° cone around forward direction (0° relative to robot)
        forward_angle = 0.0
        cone_half_width = self.cone_angle / 2
        angles = np.array([msg.angle_min + i * msg.angle_increment for i in range(len(msg.ranges))])
        cone_indices = np.where((angles >= forward_angle - cone_half_width) & 
                               (angles <= forward_angle + cone_half_width))[0]
        
        # Get valid distances in the cone
        valid_distances = [msg.ranges[i] for i in cone_indices if np.isfinite(msg.ranges[i]) and 
                          msg.range_min <= msg.ranges[i] <= msg.range_max]
        
        # Compute average distance
        self.front_distance = np.mean(valid_distances) if valid_distances else 10.0
        
    def stop_robot(self):
        # Publish zero velocity to stop the robot
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.cmd_vel_pub.publish(twist)
        # Cancel any active Nav2 goals
        if self.nav_client.server_is_ready():
            self.nav_client.cancel_goal()
        self.get_logger().info('Robot stopped')
        
    def move_forward(self):
        # Publish forward velocity for classification
        twist = Twist()
        twist.linear.x = self.forward_speed
        twist.angular.z = 0.0
        self.cmd_vel_pub.publish(twist)
        self.get_logger().info('Moving forward for classification')
        
    def process_sign(self):
        # Stop the robot before processing the sign
        self.stop_robot()
        
        if self.current_sign == 0:  # Empty wall
            self.recovery_count += 1
            if self.recovery_count < self.max_recovery_attempts:
                self.get_logger().info(f'Recovery attempt {self.recovery_count}: Rotating 90 deg')
                self.send_turn_goal(np.pi / 2)
                self.state = 'TURN'
            else:
                self.get_logger().warn('Max recovery attempts reached, stopping')
                self.state = 'IDLE'
            return
            
        self.recovery_count = 0
        if self.current_sign == 5:  # Goal
            self.get_logger().info('Goal reached!')
            msg = Bool()
            msg.data = True
            self.goal_pub.publish(msg)
            self.state = 'GOAL_REACHED'
            return
            
        # Turn based on sign
        angle = self.turn_angles[self.current_sign]
        self.send_turn_goal(angle)
        self.state = 'TURN'
        
    def send_turn_goal(self, angle):
        if self.current_pose is None:
            self.get_logger().warn('No pose available, cannot send turn goal')
            return
            
        # Ensure robot is stopped
        self.stop_robot()
        
        goal_msg = NavigateToPose.Goal()
        goal_pose = PoseStamped()
        goal_pose.header.frame_id = 'map'
        goal_pose.header.stamp = self.get_clock().now().to_msg()
        
        # Keep current position, change yaw
        goal_pose.pose.position = self.current_pose.position
        current_quat = [
            self.current_pose.orientation.x,
            self.current_pose.orientation.y,
            self.current_pose.orientation.z,
            self.current_pose.orientation.w
        ]
        current_yaw = tf_transformations.euler_from_quaternion(current_quat)[2]
        new_yaw = current_yaw + angle
        new_quat = tf_transformations.quaternion_from_euler(0, 0, new_yaw)
        
        goal_pose.pose.orientation.x = new_quat[0]
        goal_pose.pose.orientation.y = new_quat[1]
        goal_pose.pose.orientation.z = new_quat[2]
        goal_pose.pose.orientation.w = new_quat[3]
        
        goal_msg.pose = goal_pose
        self.nav_client.wait_for_server()
        self.nav_client.send_goal_async(goal_msg)
        self.get_logger().info(f'Sent turn goal: {angle} radians')
        
    def send_move_goal(self):
        if self.current_pose is None:
            self.get_logger().warn('No pose available, cannot send move goal')
            return
            
        # Get latest scan
        try:
            scan_msg = self.wait_for_message(LaserScan, '/scan', timeout=1.0)
        except:
            self.get_logger().warn('No scan data, retrying')
            self.state = 'IDLE'
            return
            
        # Compute indices for ±10° cone around forward direction
        forward_angle = 0.0
        cone_half_width = self.cone_angle / 2
        angles = np.array([scan_msg.angle_min + i * scan_msg.angle_increment 
                          for i in range(len(scan_msg.ranges))])
        cone_indices = np.where((angles >= forward_angle - cone_half_width) & 
                               (angles <= forward_angle + cone_half_width))[0]
        
        # Get valid distances in the cone
        valid_distances = [scan_msg.ranges[i] for i in cone_indices 
                          if np.isfinite(scan_msg.ranges[i]) and 
                          scan_msg.range_min <= scan_msg.ranges[i] <= scan_msg.range_max]
        if not valid_distances:
            self.get_logger().warn('No valid ranges in forward cone, retrying')
            self.state = 'IDLE'
            return
            
        # Find furthest distance and corresponding angle
        max_distance = max(valid_distances)
        max_idx = cone_indices[valid_distances.index(max_distance)]
        angle = scan_msg.angle_min + max_idx * scan_msg.angle_increment
        
        # Compute goal position in map frame
        goal_pose = PoseStamped()
        goal_pose.header.frame_id = 'map'
        goal_pose.header.stamp = self.get_clock().now().to_msg()
        
        # Transform LIDAR range to map frame
        current_yaw = tf_transformations.euler_from_quaternion([
            self.current_pose.orientation.x,
            self.current_pose.orientation.y,
            self.current_pose.orientation.z,
            self.current_pose.orientation.w
        ])[2]
        global_angle = current_yaw + angle
        goal_pose.pose.position.x = self.current_pose.position.x + max_distance * np.cos(global_angle)
        goal_pose.pose.position.y = self.current_pose.position.y + max_distance * np.sin(global_angle)
        
        # Keep current orientation
        goal_pose.pose.orientation = self.current_pose.orientation
        
        # Check if goal cell was visited
        cell_x, cell_y = int(goal_pose.pose.position.x), int(goal_pose.pose.position.y)
        if (cell_x, cell_y) in self.visited_cells:
            self.get_logger().warn('Goal cell already visited, triggering recovery')
            self.recovery_count += 1
            if self.recovery_count < self.max_recovery_attempts:
                self.send_turn_goal(np.pi / 2)
                self.state = 'TURN'
            else:
                self.get_logger().warn('Max recovery attempts reached, stopping')
                self.state = 'IDLE'
            return
            
        self.visited_cells.add((cell_x, cell_y))
        
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = goal_pose
        self.nav_client.wait_for_server()
        self.nav_client.send_goal_async(goal_msg)
        self.get_logger().info(f'Sent move goal: {max_distance}m at angle {angle}')
        
    def run(self):
        if self.state == 'IDLE':
            if self.front_distance <= self.classification_distance and self.current_pose is not None:
                # Start classification and move forward
                self.state = 'CLASSIFY'
                try:
                    sign_node = self.get_node('sign_classifier_node')
                    sign_node.start_classification()
                    self.get_logger().info('Transition to CLASSIFY')
                    self.move_forward()  # Start moving
                except:
                    self.get_logger().warn('Sign classifier node not found, retrying')
                    
        elif self.state == 'CLASSIFY':
            # Continue moving until 0.4m
            if self.front_distance > self.min_distance:
                self.move_forward()
            else:
                self.stop_robot()  # Stop at 0.4m
                self.get_logger().info('Reached 0.4m, waiting for classification')
            # Wait for sign_callback to transition to TURN or GOAL_REACHED
            
        elif self.state == 'TURN':
            # Simplified: Assume Nav2 goal completion
            self.send_move_goal()
            self.state = 'MOVE'
            self.get_logger().info('Transition to MOVE')
            
        elif self.state == 'MOVE':
            # Simplified: Assume Nav2 goal completion
            self.state = 'IDLE'
            self.get_logger().info('Transition to IDLE')
            
        elif self.state == 'GOAL_REACHED':
            self.stop_robot()  # Ensure robot stops at goal
            pass
            
def main(args=None):
    rclpy.init(args=args)
    node = NavigatorNode()
    rate = node.create_rate(10)  # 10 Hz
    while rclpy.ok():
        node.run()
        rclpy.spin_once(node)
        rate.sleep()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()