import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time
from std_msgs.msg import Float32MultiArray

from geometry_msgs.msg import PoseStamped
import tf2_ros
from tf_transformations import euler_from_quaternion
import time


class PoseListener(Node):
    def __init__(self):
        super().__init__('pose_listener')

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.timer = self.create_timer(1.0, self.timer_callback)  # 1 Hz
        # publish the map→base_link pose as a [x,y,yaw] array
        self.pose_pub = self.create_publisher(Float32MultiArray, '/pose_array', 10)

    def timer_callback(self):
        try:
            # Ask for the latest available transform
            now = self.get_clock().now()
            timeout = Duration(seconds=2.0)

            if not self.tf_buffer.can_transform('map', 'base_link', now, timeout):
                self.get_logger().warn('Transform map → base_link not yet available.')
                return

            transform = self.tf_buffer.lookup_transform('map', 'base_link', now)

            # Extract translation
            x = transform.transform.translation.x
            y = transform.transform.translation.y

            # Extract yaw from quaternion
            q = transform.transform.rotation
            _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])

            # self.get_logger().info(f"Robot position: x={x:.2f}, y={y:.2f}, yaw={yaw:.2f} rad")

        except Exception as e:
            self.get_logger().error(f"Transform lookup failed: {str(e)}")
        
        # publish it as [x, y, yaw]
        msg = Float32MultiArray()
        msg.data = [x, y, yaw]
        self.pose_pub.publish(msg)
        self.get_logger().info(f"Publishing robot position: {msg.data}")


def main(args=None):
    rclpy.init(args=args)
    node = PoseListener()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
