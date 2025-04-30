import time
import rclpy
from rclpy.node import Node
import numpy as np

from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped

class WaypointPublisher(Node):

    def __init__(self):
        # create Node
        super().__init__('waypoint_publisher')
        
        # Set up QoS Profiles
        image_qos_profile = QoSProfile(depth=10)
        image_qos_profile.history = QoSHistoryPolicy.KEEP_LAST
        image_qos_profile.durability = QoSDurabilityPolicy.VOLATILE 
        image_qos_profile.reliability = QoSReliabilityPolicy.BEST_EFFORT 
        
        # create subscriber to /navigate_to_pose/_action/feedback topic
        self.subscription = self.create_subscription(PoseWithCovarianceStamped, 
                                                     '/amcl_pose', 
                                                     self.calc_waypoint, 
                                                     image_qos_profile)
        self.subscription
        
        # create publisher to /goal_pose topic
        self.publisher_ = self.create_publisher(PoseStamped, '/goal_pose', image_qos_profile)
        
        # create timer callback
        timer_period = 0.1
        self.timer = self.create_timer(timer_period, self.timer_callback)
        
        self.goal_idx = 0
        self.threshold = 0.5

        self.goal_points = np.array([[1.60, 0.77],      # for real robot
                                     [2.54, 1.186], 
                                     [0.0, 0.0]])
        
        # self.goal_points = np.array([[1.52, 1.76],        # for sim
        #                              [2.72, -.035], 
        #                              [4.46, 1.76]])
        
    def calc_waypoint(self, currentPoseMsg):
        # Extract x, y
        x_current = currentPoseMsg.pose.pose.position.x
        y_current = currentPoseMsg.pose.pose.position.y
        
        if self.goal_idx < len(self.goal_points)-1:
            goal_x = self.goal_points[self.goal_idx, 0]
            goal_y = self.goal_points[self.goal_idx, 1]
            dist_to_goal = np.sqrt((x_current - goal_x) ** 2 + (y_current - goal_y) ** 2)
            
            if dist_to_goal < self.threshold:
                time.sleep(3)
                self.goal_idx += 1
                

    def timer_callback(self):
        pose_msg = PoseStamped()
        print("testing")
        
        # Fill in the header
        # Get current ROS time from the node's clock
        ros_time = self.get_clock().now()  # Returns rclpy.time.Time
        pose_msg.header.stamp = ros_time.to_msg()
        pose_msg.header.frame_id = 'map'        # Reference frame (our map name)

        # Position (x, y, z)
        pose_msg.pose.position.x = float(self.goal_points[self.goal_idx, 0])
        pose_msg.pose.position.y = float(self.goal_points[self.goal_idx, 1])
        pose_msg.pose.position.z = 0.0

        # # Orientation (quaternion: x, y, z, w)
        # pose_msg.pose.orientation.x = 0.0
        # pose_msg.pose.orientation.y = 0.0
        # pose_msg.pose.orientation.z = 0.0
        # pose_msg.pose.orientation.w = 1.0  # Identity quaternion (no rotation)
        
        self.publisher_.publish(pose_msg)
        self.get_logger().info(f'Publishing: {pose_msg.pose}')


def main(args=None):
    rclpy.init(args=args)

    waypoint_publisher = WaypointPublisher()

    rclpy.spin(waypoint_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    waypoint_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()