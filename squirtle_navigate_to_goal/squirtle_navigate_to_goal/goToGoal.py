# Shahmeel Naseem, Evan Dodani

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point
from geometry_msgs.msg._twist import Twist  # noqa: F401
from std_msgs.msg._float32 import Float32  # noqa: F401
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy

# etc
import numpy as np
import math
import time


class print_transformed_odom(Node):
    def __init__(self):
        super().__init__('print_fixed_odom')
        # State (for the update_Odometry code)
        self.Init = True
        self.Init_pos = Point()
        self.Init_pos.x = 0.0
        self.Init_pos.y = 0.0
        self.Init_ang = 0.0
        self.globalPos = Point()
        self.timer = self.create_timer(0.1, self.cmd_vel_timer_callback) # 0.1 second timer
        self.goal_reached_timer = None
        
        self.angular_cmd_vel = None
        self.linear_cmd_velocity = None
        self.goal_1_flag = False
        self.goal_2_flag = False
        self.goal_3_flag = False
        self.pause_flag = False
        self.add_wayPoints_flag = False
        self.obstacle_dist = None
        
        self.avoid_1_flag = False
        self.avoid_2_flag = False
        
        self.lin_err = None
        self.ang_err = None
        
        image_qos_profile = QoSProfile(depth=10)
        image_qos_profile.history = QoSHistoryPolicy.KEEP_LAST
        image_qos_profile.durability = QoSDurabilityPolicy.VOLATILE 
        image_qos_profile.reliability = QoSReliabilityPolicy.BEST_EFFORT 

        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            1)
        self.odom_sub  # prevent unused variable warning
        
        self._dist_sub = self.create_subscription(
            Float32,
            '/object_distance',
            self.avoid_obstacle,
            image_qos_profile)
        self._dist_sub
        
        self._cmd_vel_publisher = self.create_publisher(
            Twist,
            '/cmd_vel',
            10)
        self._cmd_vel_publisher

    def odom_callback(self, data):
        self.update_Odometry(data)

    def update_Odometry(self,Odom):
        position = Odom.pose.pose.position
        
        #Orientation uses the quaternion aprametrization.
        #To get the angular position along the z-axis, the following equation is required.
        q = Odom.pose.pose.orientation
        orientation = np.arctan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))

        if self.Init:
            #The initial data is stored to by subtracted to all the other values as we want to start at position (0,0) and orientation 0
            self.Init = False
            self.Init_ang = orientation
            self.globalAng = self.Init_ang
            Mrot = np.matrix([[np.cos(self.Init_ang), np.sin(self.Init_ang)],[-np.sin(self.Init_ang), np.cos(self.Init_ang)]])        
            self.Init_pos.x = Mrot.item((0,0))*position.x + Mrot.item((0,1))*position.y
            self.Init_pos.y = Mrot.item((1,0))*position.x + Mrot.item((1,1))*position.y
            self.Init_pos.z = position.z
        Mrot = np.matrix([[np.cos(self.Init_ang), np.sin(self.Init_ang)],[-np.sin(self.Init_ang), np.cos(self.Init_ang)]])        

        #We subtract the initial values
        self.globalPos.x = Mrot.item((0,0))*position.x + Mrot.item((0,1))*position.y - self.Init_pos.x
        self.globalPos.y = Mrot.item((1,0))*position.x + Mrot.item((1,1))*position.y - self.Init_pos.y
        self.globalAng = orientation - self.Init_ang
    
        self.get_logger().info('Transformed global pose is x:{}, y:{}, a:{}'.format(self.globalPos.x,self.globalPos.y,self.globalAng))
        
        self.get_goal_position()
        
    def get_goal_position(self):
        
        if self.add_wayPoints_flag and self.goal_1_flag and self.goal_2_flag and self.goal_3_flag:
            goal_points = np.array([[0.75, 0.7], [0.0, 1.4]])
            self.get_logger().info(f"AVOID GOAL 1: {goal_points[0,:]} AVOID GOAL 2: {goal_points[1,:]}")
            self.get_logger().info(f"add_wayPoints_flag: {self.add_wayPoints_flag}, avoid_1_flag: {self.avoid_1_flag}, avoid_2_flag: {self.avoid_2_flag}")
            
            if not self.avoid_1_flag:
                goal_index = 0
            else:
                goal_index = 1
                self.pause_flag = True
            
            # Compute errors
            x_difference = goal_points[goal_index, 0] - self.globalPos.x
            y_difference = goal_points[goal_index, 1] - self.globalPos.y

            theta_desired = np.arctan2(y_difference, x_difference)
            linear_error = np.hypot(x_difference, y_difference)  # Norm
            angular_error = np.arctan2(np.sin(theta_desired - self.globalAng), np.cos(theta_desired - self.globalAng))  # wrap
                
            self.get_logger().info("Linear Error: %.2f      Angular Error: %.2f" % (linear_error, angular_error))
            self.lin_err = linear_error
            self.ang_err = angular_error

            # P-Controller for Linear Velocity
            linear_gain = 0.3
            self.linear_cmd_velocity = np.clip(linear_gain * linear_error, -0.1, 0.1)

            # P-Controller for Angular Velocity
            angular_gain = 1.0
            self.angular_cmd_vel = np.clip(angular_gain * angular_error, -0.75, 0.75)
            
            if linear_error < 0.05:
                self.get_logger().info("Reached Avoidance Waypoint %d" % (goal_index + 1))
                self.add_wayPoints_flag = False
                if not self.avoid_1_flag:
                    self.avoid_1_flag = True
                    self.pause_flag = False
                elif self.avoid_1_flag and not self.avoid_2_flag:
                    self.avoid_2_flag = True
                    self.pause_flag = False
                elif self.avoid_1_flag and self.avoid_2_flag:
                    # Reset avoidance flags when final waypoint is reached
                    self.avoid_1_flag = False
                    self.avoid_2_flag = False
                    self.pause_flag = True  # Pause after avoidance complete
                    
            return

        goal_points = np.array([[1.5, 0.0], [2.5, 0.75], [1.5, 1.4], [0.0, 1.4]])
        self.get_logger().info("Goal 1 flag: %s  Goal 2 flag: %s   Goal 3 flag: %s" % (self.goal_1_flag, self.goal_2_flag, self.goal_3_flag))
        if not self.goal_1_flag:
            goal_index = 0
        elif self.goal_1_flag and not self.goal_2_flag and not self.goal_3_flag:
            goal_index = 1
        elif self.goal_1_flag and self.goal_2_flag and not self.goal_3_flag:
            goal_index = 2
        else:
            goal_index = 3
        self.get_logger().info("Goal index: %d" % goal_index)
            
        # Compute errors
        x_difference = goal_points[goal_index, 0] - self.globalPos.x
        y_difference = goal_points[goal_index, 1] - self.globalPos.y

        theta_desired = np.arctan2(y_difference, x_difference)
        linear_error = np.hypot(x_difference, y_difference)  # Norm
        angular_error = np.arctan2(np.sin(theta_desired - self.globalAng), np.cos(theta_desired - self.globalAng))  # wrap
                
        self.get_logger().info("Linear Error: %.2f      Angular Error: %.2f" % (linear_error, angular_error))
        self.lin_err = linear_error
        self.ang_err = angular_error

        # P-Controller for Linear Velocity
        linear_gain = 0.3
        self.linear_cmd_velocity = np.clip(linear_gain * linear_error, -0.1, 0.1)

        # P-Controller for Angular Velocity
        angular_gain = 1.0
        self.angular_cmd_vel = np.clip(angular_gain * angular_error, -0.75, 0.75)
                    
        if linear_error < 0.05:
            self.get_logger().info("Reached Goal %d. Waiting for 10 seconds..." % (goal_index + 1))
                    
            if not self.goal_1_flag and not self.goal_2_flag and not self.goal_3_flag:
                self.goal_1_flag = True
            elif self.goal_1_flag and not self.goal_2_flag and not self.goal_3_flag:
                self.goal_2_flag = True 
            elif self.goal_1_flag and self.goal_2_flag and not self.goal_3_flag:
                self.goal_3_flag = True

            if self.goal_1_flag and self.goal_2_flag and not self.goal_3_flag:
                self.pause_flag = False
            else:
                self.pause_flag = True  # Set the flag to pause the robot    
    
    def avoid_obstacle(self, obj_dist_msg):
        self.obstacle_dist = obj_dist_msg.data
        if self.obstacle_dist <= 0.4:
            self.add_wayPoints_flag = True
        #else:
            #self.add_wayPoints_flag = False
        self.get_logger().info('Obstacle distance: %.2f' % (self.obstacle_dist))
        
    def cmd_vel_timer_callback(self):  
            
        self.get_logger().info("Pause flag: %s" % self.pause_flag)
        if (self.angular_cmd_vel is not None) and (self.linear_cmd_velocity is not None):
            twist_msg = Twist()
            if self.pause_flag:
                twist_msg.linear.x = 0.0
                twist_msg.angular.z = 0.0
            else:
                twist_msg.linear.x = self.linear_cmd_velocity
                twist_msg.angular.z = self.angular_cmd_vel
            twist_msg.linear.y = 0.0
            twist_msg.linear.z = 0.0
            twist_msg.angular.x = 0.0
            twist_msg.angular.y = 0.0
            
            self._cmd_vel_publisher.publish(twist_msg)
            self.get_logger().info('Publishing linear velocity (x,y,z): (%.2f, %.2f, %.2f)\n Publishing angular velocity (x,y,z): (%.2f, %.2f, %.2f)' % (twist_msg.linear.x, twist_msg.linear.y, twist_msg.linear.z, twist_msg.angular.x, twist_msg.angular.y, twist_msg.angular.z)) 
            if self.pause_flag:
                self.get_logger().info('Pausing for 10 seconds')
                time.sleep(7)
                self.pause_flag = False

    
def main(args=None):
    rclpy.init(args=args)
    print_odom = print_transformed_odom()
    rclpy.spin(print_odom)
    print_odom.destroy_node()
    rclpy.shutdown()

if __name__=='__main__':
    main()