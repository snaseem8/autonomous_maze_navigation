import rclpy
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener
from rclpy.duration import Duration
from tf_transformations import euler_from_quaternion
from geometry_msgs.msg import PoseStamped


class RealRobotPose(Node):
    def __init__(self):
        super().__init__('real_robot_pose_listener')

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.timer = self.create_timer(1.0, self.get_pose)  # check every second

    def get_pose(self):
        try:
            # target frame is usually 'map' (if using AMCL), otherwise use 'odom'
            from_frame = 'map'
            to_frame = 'base_link'

            if not self.tf_buffer.can_transform(from_frame, to_frame, rclpy.time.Time()):
                self.get_logger().warn(f"No transform from {from_frame} to {to_frame} yet")
                return

            trans = self.tf_buffer.lookup_transform(
                from_frame,
                to_frame,
                rclpy.time.Time(),  # time=0 → latest transform
                timeout=Duration(seconds=2)
            )

            x = trans.transform.translation.x
            y = trans.transform.translation.y
            q = trans.transform.rotation
            _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])

            self.get_logger().info(f"Real Robot Pose → x: {x:.2f}, y: {y:.2f}, yaw: {yaw:.2f} rad")

        except Exception as e:
            self.get_logger().error(f"Transform error: {str(e)}")


def main(args=None):
    rclpy.init(args=args)
    node = RealRobotPose()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    rclpy.shutdown()


if __name__ == '__main__':
    main()
