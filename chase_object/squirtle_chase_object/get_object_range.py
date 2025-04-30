import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32MultiArray, Float32
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy
import numpy as np

class ObjectRangeNode(Node):
    def __init__(self):
        super().__init__('object_range_node')
        
        # Configure QoS for LIDAR and coordinate subscriptions
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
        
        # Subscribe to object coordinates
        self._coord_subscriber = self.create_subscription(
            Float32MultiArray,
            '/coordinates',
            self._coord_callback,
            qos_profile
        )
        
        # Publish object distance
        self._distance_publisher = self.create_publisher(Float32, '/object_distance', 10)
        
        self._coordinates = None
        self._object_distance = None
        self._timer = self.create_timer(0.1, self._timer_callback)

    def _coord_callback(self, msg):
        # Store incoming coordinates
        self._coordinates = msg.data

    def _laser_callback(self, msg):
        desired_distance = 0.5  # Target distance in meters
        
        # Process LIDAR ranges, replacing invalid values
        lidar_ranges = np.array(msg.ranges)
        lidar_ranges[np.isnan(lidar_ranges) | np.isinf(lidar_ranges)] = desired_distance
        lidar_ranges[(lidar_ranges < 0.1) | (lidar_ranges > 5)] = desired_distance
        
        if self._coordinates is not None:
            xc_angle = np.deg2rad(self._coordinates[6])  # Centroid angle in radians
            if xc_angle == 0:
                self._object_distance = desired_distance
            else:
                # Calculate index for centroid angle
                angle_increment = msg.angle_increment
                xc_index = int(np.ceil((xc_angle - msg.angle_min) / angle_increment))
                
                # Average distances around centroid (±5 indices)
                indices = (np.arange(-5, 6) + xc_index) % len(lidar_ranges)
                self._object_distance = np.mean(lidar_ranges[indices])
        else:
            self._object_distance = desired_distance

    def _timer_callback(self):
        # Publish object distance if available
        if self._coordinates is not None and self._object_distance is not None:
            distance_msg = Float32()
            distance_msg.data = float(self._object_distance)
            self._distance_publisher.publish(distance_msg)
            self.get_logger().info(f'Publishing object distance: {distance_msg.data:.2f} m')

def main():
    rclpy.init()
    node = ObjectRangeNode()
    
    try:
        rclpy.spin(node)
    except SystemExit:
        rclpy.logging.get_logger('ObjectRangeNode').info('Shutting down')
    
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
