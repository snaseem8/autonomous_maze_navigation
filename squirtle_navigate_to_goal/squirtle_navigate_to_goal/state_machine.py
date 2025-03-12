# Shahmeel Naseem, Evan Dodani

# take in both vectors: obstacle vector and goal vector.
# if obstacle distance is within a threshold AND dot product of obstacle_vec and goal_vec is positive,
# do follow wall behavior. Else go to goal

# decide whether to execute goToGoal node or avoid obstacle node.

import rclpy
from rclpy.node import Node
from sensor_msgs.msg._laser_scan import LaserScan  # noqa: F401
from std_msgs.msg._float32_multi_array import Float32MultiArray  # noqa: F401
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy

import numpy as np

# Shahmeel Naseem, Evan Dodani

import rclpy
from rclpy.node import Node
from geometry_msgs.msg._twist import Twist  # noqa: F401
from std_msgs.msg._float32_multi_array import Float32MultiArray  # noqa: F401
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy

import numpy as np

class StateMachine(Node):
    def __init__(self):        
        # Creates the node.
        super().__init__('state_machine')
    
        # Set up QoS Profiles for passing images over WiFi
        image_qos_profile = QoSProfile(depth=10)
        image_qos_profile.history = QoSHistoryPolicy.KEEP_LAST
        image_qos_profile.durability = QoSDurabilityPolicy.VOLATILE 
        image_qos_profile.reliability = QoSReliabilityPolicy.BEST_EFFORT 

        self._obstacle_subscriber = self.create_subscription(
            Float32MultiArray,
            '/object_distance',
            self._get_obstacle_callback,
            image_qos_profile)
        self._obstacle_subscriber
        
        self._goal_subscriber = self.create_subscription(
            Float32MultiArray,
            '/goal_distance',
            self._get_goal_callback,
            image_qos_profile)
        self._goal_subscriber
        
        self._cmd_vel_publisher = self.create_publisher(
            Twist,
            '/cmd_vel',
            image_qos_profile)
        self._cmd_vel_publisher

        self.timer = self.create_timer(0.1, self.timer_callback)  # Timer to publish coordinates periodically

    #def _get_obstacle_callback(self, ):
        # get the info and enter self._decide_controller callback
        
    #def _get_goal_callback(self, ):
        # get the info and enter self._decide_controller callback
        
    #def _decide_controller(self):
        # decide on node to run
        
    #def timer_callback(self):
        # publish decision to make sure correct node runs

def main():
    rclpy.init()  # Init routine needed for ROS2.
    state_machine = StateMachine()  # Create class object to be used.
    
    try:
        rclpy.spin(state_machine)  # Trigger callback processing.        
    except SystemExit:
        rclpy.logging.get_logger("Obstacle Avoider Node Info...").info("Shutting Down")
    
    # Clean up and shutdown.
    state_machine.destroy_node()  
    rclpy.shutdown()


if __name__ == '__main__':
    main()