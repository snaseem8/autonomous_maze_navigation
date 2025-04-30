import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy
import numpy as np

class ObstacleAvoiderNode(Node):
    def __init__(self):
        super().__init__('obstacle_avoider_node')

        # Configure QoS for LIDAR subscription
        qos_profile = QoSProfile(
            depth=10,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            reliability=QoSReliabilityPolicy.BEST_EFFORT
        )

        # Subscribe to LIDAR scan data
        self._lidar_subscriber = self.create_subscription(
            LaserScan,
            '/scan',
            self._laser_callback,
            qos_profile
        )

        # Publish minimum obstacle distance
        self._distance_publisher = self.create_publisher(Float32, '/object_distance', 10)

        # State variables
        self._object_distance = None
        self._timer = self.create_timer(0.1, self._publish_distance_callback)

    def _laser_callback(self, msg):
        # Process LIDAR ranges within ±30° from front
        angles = np.arange(msg.angle_min, msg.angle_max + msg.angle_increment, msg.angle_increment)
        front_mask = (angles >= np.deg2rad(-30)) & (angles <= np.deg2rad(30))
        valid_ranges = np.array(msg.ranges)[front_mask]
        valid_ranges = valid_ranges[np.isfinite(valid_ranges)]

        self._object_distance = float(np.min(valid_ranges)) if valid_ranges.size > 0 else None

    def _publish_distance_callback(self):
        # Publish minimum obstacle distance
        if self._object_distance is not None:
            msg = Float32()
            msg.data = self._object_distance
            self._distance_publisher.publish(msg)
            self.get_logger().info(f'Obstacle distance: {msg.data:.2f} m')
        else:
            self.get_logger().info('No obstacle detected')

def main():
    rclpy.init()
    node = ObstacleAvoiderNode()
    try:
        rclpy.spin(node)
    except (SystemExit, KeyboardInterrupt):
        node.get_logger().info('Shutting down')
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
