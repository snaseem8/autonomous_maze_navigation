import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32, Float32, Float32MultiArray
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy
import numpy as np
import math
import time

class NavigatorNode(Node):
    def __init__(self):
        super().__init__('navigator_node')

        # Configure QoS for LIDAR
        lidar_qos = QoSProfile(
            depth=10,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            reliability=QoSReliabilityPolicy.BEST_EFFORT
        )

        # Configure QoS for AMCL (reliable for pose data)
        amcl_qos = QoSProfile(
            depth=5,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE
        )

        # Subscribers
        self._class_sub = self.create_subscription(Int32, '/sign_class', self._class_callback, 10)
        self._scan_sub = self.create_subscription(LaserScan, '/scan', self._scan_callback, lidar_qos)
        self._amcl_sub = self.create_subscription(PoseWithCovarianceStamped, '/amcl_pose', self._amcl_callback, amcl_qos)
        self._centroid_sub = self.create_subscription(Float32MultiArray, '/coordinates', self._coordinate_callback, lidar_qos)

        # Publishers
        self._cmd_vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self._front_dist_publisher = self.create_publisher(Float32, '/front_dist', 10)

        # Control timer
        self._controller_timer = self.create_timer(0.1, self._controller_timer_callback)

        # Navigation parameters
        self._cone_angle = np.deg2rad(14.0)
        self._target_distance = 0.5
        self._linear_speed_max = 0.15
        self._angular_speed_max = 1.5
        self._kp_linear = 0.4
        self._kp_angular = 1.0
        self._kp_forward_angular = 0.5
        self._angular_tolerance = np.deg2rad(5.0)
        self._wall_alignment_fov = np.deg2rad(30.0)
        self._kp_wall_alignment = 1.0
        self._wall_alignment_tolerance = np.deg2rad(3.0)
        self._stabilization_time = 1.5
        self._post_alignment_time = 1.5
        self._vote_threshold = 2

        # State variables
        self._state = 'IDLE'
        self._front_distance = float('inf')
        self._current_class = None
        self._last_class = None
        self._class_votes = []
        self._goal_reached = False
        self._current_pose = (0.0, 0.0, 0.0)  # (x, y, yaw)
        self._target_yaw = None
        self._ignore_classification = False
        self._initial_class_done = False
        self._turn_complete_time = None
        self._alignment_complete_time = None

        # Perform startup wiggle for AMCL initialization
        self._startup_wiggle()

    def _startup_wiggle(self):
        # Perform left-right wiggle to aid AMCL
        self.get_logger().info('Starting AMCL wiggle')
        cmd = Twist()

        for angular_vel in [0.5, -0.5, 0.0]:  # Left, right, stop
            cmd.angular.z = angular_vel
            start_time = time.time()
            while time.time() - start_time < 1.0:
                self._cmd_vel_publisher.publish(cmd)
                time.sleep(0.1)

        self.get_logger().info('Wiggle complete')

    def _quaternion_to_yaw(self, quaternion):
        # Convert quaternion to yaw angle
        x, y, z, w = quaternion
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)

    def _normalize_angle(self, angle):
        # Normalize angle to [-pi, pi]
        return math.atan2(math.sin(angle), math.cos(angle))

    def _coordinate_callback(self, msg):
        self._coord = msg.data

    def _scan_callback(self, msg):
        # Calculate front distance within 14° cone
        angles = np.array([msg.angle_min + i * msg.angle_increment for i in range(len(msg.ranges))])
        cone_indices = np.where((angles >= -self._cone_angle / 2) & (angles <= self._cone_angle / 2))[0]
        valid_distances = [msg.ranges[i] for i in cone_indices if np.isfinite(msg.ranges[i]) and
                           msg.range_min <= msg.ranges[i] <= msg.range_max]
        self._front_distance = np.mean(valid_distances) if valid_distances else float('inf')

        # Publish front distance
        dist_msg = Float32()
        dist_msg.data = self._front_distance
        self._front_dist_publisher.publish(dist_msg)

        # Handle initial classification at wall
        if not self._initial_class_done and self._state == 'IDLE' and self._front_distance <= self._target_distance:
            if self._last_class is None:
                self.get_logger().warn('Initial wall detected but no classifier message')
            else:
                self._current_class = self._last_class
                self._initial_class_done = True
                self.get_logger().info(f'Initial classification: {self._current_class}')
                self._process_sign()
            return

        # Check for wall proximity in forward state
        if self._state == 'MOVING_FORWARD' and self._front_distance <= self._target_distance:
            self._stop_robot()
            self._state = 'ALIGNING'
            self._wall_scan = msg
            self.get_logger().info('Reached wall, starting alignment')
            return

        # Store scan data for alignment
        if self._state == 'ALIGNING':
            self._wall_scan = msg

    def _class_callback(self, msg):
        # Store latest classification
        self._last_class = msg.data

        # Collect votes during approach
        if self._state == 'MOVING_FORWARD' and 0.4 < self._front_distance <= 1.0:
            self._class_votes.append(msg.data)
            self.get_logger().info(f'Collected vote: {msg.data}')

        # Process sign in IDLE state
        if self._state == 'IDLE' and not self._goal_reached and not self._ignore_classification:
            self._current_class = msg.data
            self.get_logger().info(f'Received sign class: {self._current_class}')
            self._process_sign()

    def _amcl_callback(self, msg):
        try:
            # Extract pose data
            x = msg.pose.pose.position.x
            y = msg.pose.pose.position.y
            yaw = self._quaternion_to_yaw([msg.pose.pose.orientation.x, msg.pose.pose.orientation.y,
                                          msg.pose.pose.orientation.z, msg.pose.pose.orientation.w])
            self._current_pose = (x, y, yaw)

            # Log pose updates
            if any(v != 0.0 for v in [x, y, yaw]):
                self.get_logger().info(f'AMCL pose: x={x:.3f}, y={y:.3f}, yaw={yaw:.3f}')
        except Exception as e:
            self.get_logger().error(f'AMCL callback error: {e}')

    def _controller_timer_callback(self):
        # Handle state-specific control logic
        current_time = self.get_clock().now().seconds_nanoseconds()[0]

        if self._state == 'TURNING' and self._target_yaw is not None:
            self._turn_controller()
        elif self._state == 'STABILIZING' and self._turn_complete_time is not None:
            if current_time - self._turn_complete_time >= self._stabilization_time:
                self._state = 'MOVING_FORWARD'
                self.get_logger().info('Camera stabilized, moving forward')
        elif self._state == 'MOVING_FORWARD' and self._target_yaw is not None:
            self._drive_controller()
        elif self._state == 'ALIGNING':
            self._wall_alignment_controller()
        elif self._state == 'ALIGNMENT_STABILIZING' and self._alignment_complete_time is not None:
            if current_time - self._alignment_complete_time >= self._post_alignment_time:
                self._state = 'IDLE'
                self._target_yaw = None
                self._ignore_classification = False
                self.get_logger().info('Alignment complete, processing sign')
                self._current_class = max(set(self._class_votes), key=self._class_votes.count) if len(self._class_votes) >= self._vote_threshold else self._last_class
                self._class_votes.clear()
                self._process_sign()

    def _process_sign(self):
        # Process sign classification and set turn direction
        if self._current_class is None or self._goal_reached or self._current_pose is None:
            return

        self._state = 'TURNING'
        self._ignore_classification = True
        _, _, current_yaw = self._current_pose

        # Sign classes:
        if self._current_class == 5:  # Goal
            self._target_yaw = current_yaw - np.pi / 2  # Turn right
            self.get_logger().info(f'Goal detected, turning right to yaw {self._target_yaw:.2f}')
        elif self._current_class == 0:  # Empty (recovery)
            self._target_yaw = current_yaw - np.pi / 2  # Turn right
            self.get_logger().info(f'Recovery, turning right to yaw {self._target_yaw:.2f}')
        elif self._current_class == 1:  # Left
            self._target_yaw = current_yaw + np.pi / 2  # Turn left
            self.get_logger().info(f'Turning left to yaw {self._target_yaw:.2f}')
        elif self._current_class == 2:  # Right
            self._target_yaw = current_yaw - np.pi / 2  # Turn right
            self.get_logger().info(f'Turning right to yaw {self._target_yaw:.2f}')
        elif self._current_class in [3, 4]:  # Do not enter or Stop
            self._target_yaw = current_yaw + np.pi  # Turn around
            self.get_logger().info(f'Turning around to yaw {self._target_yaw:.2f}')
        else:
            self.get_logger().error(f'Unknown sign class: {self._current_class}')
            self._state = 'IDLE'
            self._target_yaw = None
            self._ignore_classification = False
            return

        self._target_yaw = self._normalize_angle(self._target_yaw)

    def _turn_controller(self):
        # Control robot to reach target yaw
        if self._current_pose is None or self._target_yaw is None:
            return

        _, _, current_yaw = self._current_pose
        angular_error = np.arctan2(np.sin(self._target_yaw - current_yaw), np.cos(self._target_yaw - current_yaw))
        angular_vel = self._kp_angular * angular_error
        angular_vel = max(min(angular_vel, self._angular_speed_max), -self._angular_speed_max)

        cmd = Twist()
        cmd.angular.z = angular_vel if abs(angular_error) > self._angular_tolerance else 0.0
        self._cmd_vel_publisher.publish(cmd)

        if abs(angular_error) <= self._angular_tolerance:
            self._stop_robot()
            self._state = 'STABILIZING'
            self._turn_complete_time = self.get_clock().now().seconds_nanoseconds()[0]
            self.get_logger().info('Turn complete, stabilizing')

    def _drive_controller(self):
        # Control robot to move forward while maintaining heading
        if self._current_pose is None or self._target_yaw is None:
            self.get_logger().warn('Drive controller missing pose or yaw')
            return

        if self._front_distance > self._target_distance:
            linear_vel = min(self._kp_linear * (self._front_distance - (self._target_distance - 0.1)), self._linear_speed_max)
            _, _, current_yaw = self._current_pose
            yaw_error = self._normalize_angle(self._target_yaw - current_yaw)
            angular_vel = max(min(self._kp_forward_angular * yaw_error, self._angular_speed_max), -self._angular_speed_max)

            cmd = Twist()
            cmd.linear.x = linear_vel
            cmd.angular.z = angular_vel
            self._cmd_vel_publisher.publish(cmd)
            self.get_logger().info(f'Driving: dist={self._front_distance:.2f}m, yaw_err={yaw_error:.2f}rad')

    def _wall_alignment_controller(self):
        # Align robot perpendicular to wall
        if not hasattr(self, '_wall_scan'):
            self.get_logger().warn('No wall scan data for alignment')
            self._state = 'IDLE'
            return

        angles = np.array([self._wall_scan.angle_min + i * self._wall_scan.angle_increment for i in range(len(self._wall_scan.ranges))])
        fov_indices = np.where((angles >= -self._wall_alignment_fov / 2) & (angles <= self._wall_alignment_fov / 2))[0]

        min_distance = float('inf')
        min_angle = 0.0
        for i in fov_indices:
            if np.isfinite(self._wall_scan.ranges[i]) and self._wall_scan.range_min <= self._wall_scan.ranges[i] <= self._wall_scan.range_max:
                if self._wall_scan.ranges[i] < min_distance:
                    min_distance = self._wall_scan.ranges[i]
                    min_angle = angles[i]

        if min_distance < float('inf'):
            if abs(min_angle) <= self._wall_alignment_tolerance:
                self._stop_robot()
                self._alignment_complete_time = self.get_clock().now().seconds_nanoseconds()[0]
                self._state = 'ALIGNMENT_STABILIZING'
                self.get_logger().info('Wall alignment complete')
            else:
                angular_vel = max(min(-self._kp_wall_alignment * min_angle, self._angular_speed_max / 2), -self._angular_speed_max / 2)
                cmd = Twist()
                cmd.angular.z = angular_vel
                self._cmd_vel_publisher.publish(cmd)
                self.get_logger().info(f'Aligning: angle_err={min_angle:.3f}, cmd={angular_vel:.3f}')
        else:
            self.get_logger().warn('No valid wall points for alignment')
            self._state = 'IDLE'
            self._process_sign()

    def _stop_robot(self):
        # Stop robot motion
        self._cmd_vel_publisher.publish(Twist())

def main(args=None):
    rclpy.init(args=args)
    node = NavigatorNode()
    try:
        rclpy.spin(node)
    except SystemExit:
        node.get_logger().info('Shutting down')
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
