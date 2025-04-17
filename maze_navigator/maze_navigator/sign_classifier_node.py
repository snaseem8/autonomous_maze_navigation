#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from std_msgs.msg import Int32
from cv_bridge import CvBridge
import cv2
import numpy as np
from collections import Counter

# IMPORT CLASSIFIER

class SignClassifierNode(Node):
    def __init__(self):
        super().__init__('sign_classifier_node')
        self.bridge = CvBridge()
        self.model = # ADD MODEL
        
        # Parameters
        self.classification_distance = 0.7  # Meters to wall for classification
        self.min_distance = 0.4  # Stop classifying if closer
        self.num_samples = 10  # Number of images to classify
        self.confidence_threshold = 0.6  # Minimum fraction for majority vote
        
        # Subscribers
        self.image_sub = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10)
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10)
        
        # Publisher
        self.class_pub = self.create_publisher(Int32, '/sign_class', 10)
        
        # State
        self.current_image = None
        self.front_distance = float('inf')
        self.classifying = False
        self.classifications = []
        
    def image_callback(self, msg):
        self.current_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        
    def scan_callback(self, msg):
        # Compute indices for ±10° cone around forward direction (0° relative to robot)
        forward_angle = 0.0  # Forward direction in robot frame
        cone_half_width = self.cone_angle / 2
        angles = np.array([msg.angle_min + i * msg.angle_increment for i in range(len(msg.ranges))])
        cone_indices = np.where((angles >= forward_angle - cone_half_width) & 
                               (angles <= forward_angle + cone_half_width))[0]
        
        # Get valid distances in the cone
        valid_distances = [msg.ranges[i] for i in cone_indices if np.isfinite(msg.ranges[i]) and 
                          msg.range_min <= msg.ranges[i] <= msg.range_max]
        
        # Compute average distance (or max if preferred)
        self.front_distance = np.mean(valid_distances) if valid_distances else 10.0
        
        if self.classifying:
            self.process_classification()
            
    def start_classification(self):
        if self.front_distance <= self.classification_distance and self.front_distance > self.min_distance:
            self.classifying = True
            self.classifications = []
            self.get_logger().info('Starting sign classification')
        else:
            self.get_logger().warn(f'Not at correct distance: {self.front_distance}m')
            
    def process_classification(self):
        if not self.classifying or self.current_image is None:
            return
            
        if self.front_distance <= self.min_distance:
            self.finish_classification()
            return
            
        # Classify current image
        pred = # ADD PREDICTION METHOD OF MODEL
        self.classifications.append(pred)
        
        if len(self.classifications) >= self.num_samples:
            self.finish_classification()
            
    def finish_classification(self):
        self.classifying = False
        if not self.classifications:
            self.get_logger().warn('No classifications collected')
            return
            
        # Compute majority vote
        counter = Counter(self.classifications)
        most_common, count = counter.most_common(1)[0]
        confidence = count / len(self.classifications)
        
        if confidence >= self.confidence_threshold:
            msg = Int32()
            msg.data = most_common
            self.class_pub.publish(msg)
            self.get_logger().info(f'Published sign class: {most_common} (confidence: {confidence:.2f})')
        else:
            self.get_logger().warn(f'Low confidence: {confidence:.2f}, discarding')
            
def main(args=None):
    rclpy.init(args=args)
    node = SignClassifierNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()