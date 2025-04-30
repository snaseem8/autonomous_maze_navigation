import rclpy
from rclpy.node import Node
import numpy as np
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
import time

class WaypointPublisherNode(Node):
    def __init__(self):
        super().__init__('waypoint_publisher_node')

        # Configure QoS for AMCL pose subscription
        qos_profile = QoSProfile(
            depth=5,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            reliability=QoSReliabilityPolicy.RELIABLE
        )

        # Subscribe to AMCL pose
        self._subscription = self.create_subscription(
            PoseWithCovarianceStamped,
            '/amcl_pose',
            self._calc_waypoint,
            qos_profile
        )

        # Publish goal poses
        self._publisher = self.create_publisher(PoseStamped, '/goal_pose', 5)

        # Timer for publishing waypoints
        self._timer = self.create_timer(0.1, self._publish_waypoint)

        # Waypoint parameters
        self._goal_index = 0
        self._threshold = 0.5
        self._pause_duration = 3.0  # Seconds to pause after reaching a goal
        self._pause_start = None
        self._goal_points = np.array([
            [0.82, 0.74],  # Waypoint 1
            [1.52, 0.75],  # Waypoint 2
            [2.33, 0.09]   # Waypoint 3
        ])

    def _calc_waypoint(self, msg):
        # Check distance to current goal and advance if within threshold
        if self._goal_index >= len(self._goal_points) or self._pause_start is not None:
            return

        x_current = msg.pose.pose.position.x
        y_current = msg.pose.pose.position.y
        goal_x, goal_y = self._goal_points[self._goal_index]
        distance = np.sqrt((x_current - goal_x) ** 2 + (y_current - goal_y) ** 2)

        if distance < self._threshold:
            self.get_logger().info(f'Reached waypoint {self._goal_index + 1}: [{goal_x:.2f}, {goal_y:.2f}]')
            self._pause_start = time.time()
            self._goal_index += 1

    def _publish_waypoint(self):
        # Publish current goal pose
        if self._goal_index >= len(self._goal_points):
            return

        # Handle pause after reaching a goal
        if self._pause_start is not None:
            if time.time() - self._pause_start < self._pause_duration:
                return
            self._pause_start = None

        # Create and publish pose message
        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = 'map'
        pose.pose.position.x = float(self._goal_points[self._goal_index, 0])
        pose.pose.position.y = float(self._goal_points[self._goal_index, 1])
        pose.pose.position.z = 0.0
        pose.pose.orientation.w = 1.0  # Identity quaternion (no rotation)

        self._publisher.publish(pose)
        self.get_logger().info(f'Publishing waypoint {self._goal_index + 1}: [{pose.pose.position.x:.2f}, {pose.pose.position.y:.2f}]')

def main():
    rclpy.init()
    node = WaypointPublisherNode()
    try:
        rclpy.spin(node)
    except (SystemExit, KeyboardInterrupt):
        node.get_logger().info('Shutting down')
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
