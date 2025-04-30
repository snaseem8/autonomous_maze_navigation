import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Twist
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy
import numpy as np

class RobotRotatorNode(Node):
    def __init__(self):
        super().__init__('robot_rotator_node')

        # Configure QoS for subscription
        qos_profile = QoSProfile(
            depth=10,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            reliability=QoSReliabilityPolicy.BEST_EFFORT
        )

        # Subscribe to object centroid coordinates
        self._centroid_subscriber = self.create_subscription(
            Float32MultiArray,
            '/coordinates',
            self._cmd_vel_callback,
            qos_profile
        )

        # Publish velocity commands
        self._cmd_vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10)

        self._angular_velocity = 0.0
        self._timer = self.create_timer(0.1, self._publish_velocity)

    def _cmd_vel_callback(self, msg):
        # Calculate angular velocity to align with object centroid
        centroid = msg.data
        if len(centroid) >= 4:
            error = centroid[0] - centroid[2]  # x_centroid - x_center
            self._angular_velocity = -0.005 * error if abs(error) > 30 else 0.0
        else:
            self._angular_velocity = 0.0

    def _publish_velocity(self):
        # Publish angular velocity command
        twist = Twist()
        twist.angular.z = self._angular_velocity
        self._cmd_vel_publisher.publish(twist)
        self.get_logger().info(f'Publishing velocity: linear=({twist.linear.x:.2f}, {twist.linear.y:.2f}, {twist.linear.z:.2f}), angular=({twist.angular.x:.2f}, {twist.angular.y:.2f}, {twist.angular.z:.2f})')

def main():
    rclpy.init()
    node = RobotRotatorNode()
    try:
        rclpy.spin(node)
    except SystemExit:
        node.get_logger().info('Shutting down')
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
