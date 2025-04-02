import time
import rclpy
from rclpy.node import Node

from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action._navigate_to_pose import NavigateToPose_FeedbackMessage

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
        self.subscription = self.create_subscription(NavigateToPose_FeedbackMessage, 
                                                     '/navigate_to_pose/_action/feedback', 
                                                     self.calc_waypoint, 
                                                     image_qos_profile)
        self.subscription
        
        # create publisher to /goal_pose topic
        self.publisher_ = self.create_publisher(PoseStamped, '/goal_pose', image_qos_profile)
        
        # create timer callback
        timer_period = 0.1
        self.timer = self.create_timer(timer_period, self.timer_callback)
        
    def calc_waypoint(self, currentPoseMsg):
        currentPose = currentPoseMsg.data
        self.get_logger().info(f'Current Pose: {currentPose}')
        print (currentPose)
        
        
        # self.x_goal = 
        # self.y_goal = 

    def timer_callback(self):
        pose_msg = PoseStamped()
        print("testing")
        
        # Fill in the header
        # pose_msg.header.stamp = time.time()  # Current time
        # pose_msg.header.frame_id = "map"          # Reference frame (our map name)

        # Position (x, y, z)
        # pose_msg.pose.position.x = self.x_goal
        # pose_msg.pose.position.y = self.y_goal
        # pose_msg.pose.position.z = 0.0

        # # Orientation (quaternion: x, y, z, w)
        # pose_msg.pose.orientation.x = 0.0
        # pose_msg.pose.orientation.y = 0.0
        # pose_msg.pose.orientation.z = 0.0
        # pose_msg.pose.orientation.w = 1.0  # Identity quaternion (no rotation)
        
        # self.publisher_.publish(pose_msg)
        # self.get_logger().info('Publishing: "%s"' % pose_msg.data)


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