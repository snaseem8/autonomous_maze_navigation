import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point, Twist
from std_msgs.msg import Float32
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy
import numpy as np
import math
import time

class GoToGoalNode(Node):
    def __init__(self):
        super().__init__('go_to_goal_node')

        # Configure QoS for subscriptions
        qos_profile = QoSProfile(
            depth=10,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            reliability=QoSReliabilityPolicy.BEST_EFFORT
        )

        # Subscribers
        self._odom_subscriber = self.create_subscription(
            Odometry, '/odom', self._odom_callback, 1)
        self._dist_subscriber = self.create_subscription(
            Float32, '/object_distance', self._avoid_obstacle_callback, qos_profile)

        # Publisher
        self._cmd_vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10)

        # Timer for velocity commands
        self._timer = self.create_timer(0.1, self._cmd_vel_callback)

        # State variables
        self._init_pose = True
        self._init_position = Point()
        self._init_angle = 0.0
        self._global_position = Point()
        self._global_angle = 0.0
        self._linear_velocity = 0.0
        self._angular_velocity = 0.0
        self._obstacle_distance = float('inf')
        self._pause_flag = False
        self._pause_start_time = None
        self._goal_flags = [False, False, False]  # Goal 1, 2, 3
        self._avoid_flags = [False, False]  # Avoid 1, 2
        self._add_waypoints = False

    def _odom_callback(self, msg):
        # Update global pose from odometry
        self._update_odometry(msg)

    def _update_odometry(self, msg):
        position = msg.pose.pose.position
        q = msg.pose.pose.orientation
        orientation = np.arctan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))

        if self._init_pose:
            self._init_pose = False
            self._init_angle = orientation
            self._init_position.x = position.x
            self._init_position.y = position.y
            self._init_position.z = position.z

        # Transform position to global frame
        cos_init = np.cos(self._init_angle)
        sin_init = np.sin(self._init_angle)
        self._global_position.x = cos_init * (position.x - self._init_position.x) + sin_init * (position.y - self._init_position.y)
        self._global_position.y = -sin_init * (position.x - self._init_position.x) + cos_init * (position.y - self._init_position.y)
        self._global_angle = orientation - self._init_angle

        self.get_logger().info(f'Global pose: x={self._global_position.x:.2f}, y={self._global_position.y:.2f}, angle={self._global_angle:.2f}')

        # Compute goal errors
        self._compute_goal_errors()

    def _compute_goal_errors(self):
        # Define goal points
        if self._add_waypoints and all(self._goal_flags):
            goal_points = np.array([[0.75, 0.7], [0.0, 1.4]])  # Avoidance waypoints
            goal_index = 0 if not self._avoid_flags[0] else 1
            self.get_logger().info(f'Avoidance goal {goal_index + 1}: {goal_points[goal_index]}')
        else:
            goal_points = np.array([[1.5, 0.0], [2.5, 0.75], [1.5, 1.4], [0.0, 1.4]])  # Main goals
            goal_index = sum(self._goal_flags)  # 0, 1, 2, or 3
            self.get_logger().info(f'Goal {goal_index + 1}: {goal_points[goal_index]}')

        # Compute errors
        x_diff = goal_points[goal_index, 0] - self._global_position.x
        y_diff = goal_points[goal_index, 1] - self._global_position.y
        linear_error = np.hypot(x_diff, y_diff)
        theta_desired = np.arctan2(y_diff, x_diff)
        angular_error = np.arctan2(np.sin(theta_desired - self._global_angle), np.cos(theta_desired - self._global_angle))

        self.get_logger().info(f'Linear error: {linear_error:.2f}, Angular error: {angular_error:.2f}')

        # P-Controller
        self._linear_velocity = np.clip(0.3 * linear_error, -0.1, 0.1)
        self._angular_velocity = np.clip(1.0 * angular_error, -0.75, 0.75)

        # Check if goal reached
        if linear_error < 0.05:
            self.get_logger().info(f'Reached {"avoidance waypoint" if self._add_waypoints else "goal"} {goal_index + 1}')
            if self._add_waypoints and all(self._goal_flags):
                self._avoid_flags[goal_index] = True
                self._add_waypoints = goal_index < 1  # Continue until last avoidance point
                self._pause_flag = not self._add_waypoints
            else:
                self._goal_flags[goal_index] = True
                self._pause_flag = not (self._goal_flags[0] and self._goal_flags[1] and not self._goal_flags[2])
            if self._pause_flag:
                self._pause_start_time = time.time()
                self.get_logger().info('Pausing for 7 seconds')

    def _avoid_obstacle_callback(self, msg):
        # Update obstacle distance and trigger avoidance waypoints
        self._obstacle_distance = msg.data
        self._add_waypoints = self._obstacle_distance <= 0.4 and all(self._goal_flags)
        self.get_logger().info(f'Obstacle distance: {self._obstacle_distance:.2f}')

    def _cmd_vel_callback(self):
        # Publish velocity commands
        if self._pause_flag and self._pause_start_time:
            if time.time() - self._pause_start_time >= 7:
                self._pause_flag = False
                self._pause_start_time = None
            twist = Twist()  # Zero velocity during pause
        else:
            twist = Twist()
            twist.linear.x = self._linear_velocity
            twist.angular.z = self._angular_velocity

        self._cmd_vel_publisher.publish(twist)
        self.get_logger().info(f'Velocity: linear={twist.linear.x:.2f}, angular={twist.angular.z:.2f}')

def main():
    rclpy.init()
    node = GoToGoalNode()
    try:
        rclpy.spin(node)
    except (SystemExit, KeyboardInterrupt):
        node.get_logger().info('Shutting down')
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
