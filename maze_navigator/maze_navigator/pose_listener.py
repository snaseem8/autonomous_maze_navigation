import rclpy
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener
from rclpy.duration import Duration
from geometry_msgs.msg import PoseStamped
from transforms3d.quaternions import quat2mat
import numpy as np

class PoseListenerNode(Node):
    def __init__(self):
        super().__init__('pose_listener_node')

        # Initialize TF2 buffer and listener
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        # Set up timer to check pose at 2 Hz
        self._timer = self.create_timer(0.5, self._pose_callback)

    def _pose_callback(self):
        # Look up transform from map to base_link
        try:
            trans = self._tf_buffer.lookup_transform(
                'map',
                'base_link',
                rclpy.time.Time(),
                timeout=Duration(seconds=2)
            )

            # Extract position and orientation
            x = trans.transform.translation.x
            y = trans.transform.translation.y
            quat = [trans.transform.rotation.w, trans.transform.rotation.x,
                    trans.transform.rotation.y, trans.transform.rotation.z]
            rot_mat = quat2mat(quat)
            yaw = np.arctan2(rot_mat[1, 0], rot_mat[0, 0])

            self.get_logger().info(f'Pose: x={x:.2f}, y={y:.2f}, yaw={yaw:.2f} rad')

        except Exception as e:
            self.get_logger().warn(f'Transform error: {str(e)}')

def main():
    rclpy.init()
    node = PoseListenerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
