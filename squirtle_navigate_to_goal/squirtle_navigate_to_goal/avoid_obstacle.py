# Shahmeel Naseem, Evan Dodani

import rclpy
from rclpy.node import Node
from sensor_msgs.msg._laser_scan import LaserScan  # noqa: F401
from std_msgs.msg._float32 import Float32  # noqa: F401
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy
from geometry_msgs.msg._twist import Twist  # noqa: F401

import numpy as np

class ObstacleAvoider(Node):
    def __init__(self):        
        super().__init__('obstacle_avoider')
    
        image_qos_profile = QoSProfile(
            depth=10,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            reliability=QoSReliabilityPolicy.BEST_EFFORT
        )

        self._LIDAR_subscriber = self.create_subscription(
            LaserScan,
            '/scan',
            self._laser_callback,
            image_qos_profile)
        
        self._dist_publisher = self.create_publisher(
            Float32,
            '/object_distance',
            image_qos_profile)

        self.avoid_obstacle_flag = False
        self.object_dist = None
        self.obstacle_pub_timer = self.create_timer(0.1, self.pub_obstacle_pos_timer_callback)

    def _laser_callback(self, lidar_msg):
        min_angle = lidar_msg.angle_min
        max_angle = lidar_msg.angle_max
        angle_increment = lidar_msg.angle_increment
        
        ranges = np.array(lidar_msg.ranges)
        angles = min_angle + np.arange(len(ranges)) * angle_increment
        ignore_angle_mask = (angles >= np.deg2rad(30)) & (angles <= np.deg2rad(330))         
        selected_ranges = ranges[~ignore_angle_mask]
        filtered_ranges = selected_ranges[np.isfinite(selected_ranges)]
        
        if len(filtered_ranges) > 0:
            self.object_dist = float(np.min(filtered_ranges))
        else:
            self.object_dist = None
    
    def pub_obstacle_pos_timer_callback(self):
        object_dist_msg = Float32()
        if self.object_dist is not None:
            object_dist_msg.data = float(self.object_dist)
            self._dist_publisher.publish(object_dist_msg)
            self.get_logger().info(str(object_dist_msg.data))
        else:
            self.get_logger().info('No obstacle detected')

def main():
    rclpy.init()
    obstacle_avoider = ObstacleAvoider()
    
    try:
        rclpy.spin(obstacle_avoider)
    except (SystemExit, KeyboardInterrupt):
        rclpy.logging.get_logger("Obstacle Avoider").info("Shutting Down")
    
    obstacle_avoider.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()