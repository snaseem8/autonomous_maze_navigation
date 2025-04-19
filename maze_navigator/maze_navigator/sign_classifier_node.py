#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, LaserScan
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy
from std_msgs.msg import Int32, Float32
from cv_bridge import CvBridge
import cv2
import os
import numpy as np
from maze_navigator.model import initialize_model, predict

class SignClassifierNode(Node):
    def __init__(self):
        super().__init__('sign_classifier_node')
        
        # Set up QoS Profiles
        image_qos_profile = QoSProfile(depth=10)
        image_qos_profile.history = QoSHistoryPolicy.KEEP_LAST
        image_qos_profile.durability = QoSDurabilityPolicy.VOLATILE 
        image_qos_profile.reliability = QoSReliabilityPolicy.BEST_EFFORT 
        
        #Set up QoS Profiles for passing images over WiFi
        image_qos_profile_img = QoSProfile(depth=5)
        image_qos_profile_img.history = QoSHistoryPolicy.KEEP_LAST
        image_qos_profile_img.durability = QoSDurabilityPolicy.VOLATILE 
        image_qos_profile_img.reliability = QoSReliabilityPolicy.BEST_EFFORT 
        
        # # Set Parameters to show image
        # self.declare_parameter('show_image_bool', True)
        # self.declare_parameter('window_name', "Raw Image")
        # self.current_image = None

        # # Determine Window Showing Based on Input
        # self._display_image = bool(self.get_parameter('show_image_bool').value)

        # # Declare some variables
        # self._titleOriginal = self.get_parameter('window_name').value
        # if self._display_image:
        #     cv2.namedWindow(self._titleOriginal, cv2.WINDOW_AUTOSIZE)
        #     cv2.moveWindow(self._titleOriginal, 50, 50)
        
        # Load model
        model_path = os.path.join(os.path.dirname(__file__), 'knn_model_color.pkl')
        try:
            self.model = initialize_model(model_path)
        except Exception as e:
            self.get_logger().error(f"Failed to initialize model: {str(e)}")
            raise
            
        # Parameters
        self.classification_distance = 1.0
        self.min_distance = 0.2  # Stop classifying if closer
        self.cone_angle = np.deg2rad(14.0)  # 14 deg cone for distance averaging
        
        # Subscribers
        self.image_sub = self.create_subscription(
            CompressedImage, '/image_raw/compressed', self.image_callback, image_qos_profile_img)
        self.front_dist_sub = self.create_subscription(
            Float32, '/front_dist', self.front_dist_callback, 10)
        
        # Publisher
        self.class_pub = self.create_publisher(Int32, '/sign_class', 10)
        
        # State
        self.current_image = None
        self.front_distance = float('inf')
        
    def image_callback(self, CompressedImage):
        try:
            print('Testing')
            # Convert compressed image to OpenCV BGR format
            img = CvBridge().compressed_imgmsg_to_cv2(CompressedImage, "bgr8")
            
            # Convert to grayscale for edge detection
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Perform Canny edge detection
            edges = cv2.Canny(blurred, 50, 150)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Filter contours by area (e.g., min area to avoid noise, adjust#> adjust as needed
                min_contour_area = 3000
                valid_contours = None # [c for c in contours if cv2.contourArea(c) > min_contour_area]
                
                if valid_contours:
                    # Get the largest valid contour (assumed to be the sign)
                    largest_contour = max(valid_contours, key=cv2.contourArea)
                    
                    # Get bounding box around the sign
                    x, y, w, h = cv2.boundingRect(largest_contour)
                    
                    # Add padding to the bounding box
                    padding = 20
                    x = max(x - padding, 0)
                    y = max(y - padding, 0)
                    w = min(w + 2 * padding, img.shape[1] - x)
                    h = min(h + 2 * padding, img.shape[0] - y)
                    
                    # Crop the image around the sign (in BGR format for classifier)
                    self.current_image = img[y:y+h, x:x+w]
                else:
                    # Fallback: use the entire image if no valid contours are found
                    # self.get_logger().warn("No valid contours detected, using entire image")
                    self.current_image = img
                    
            # # Display the processed frame for debugging
            # cv2.imshow(self._titleOriginal, self.current_image)
            
            # # Wait for key press and close window
            # if cv2.waitKey(1) == ord('q'):
            #     cv2.destroyAllWindows()  # Close all OpenCV windows when 'q' is pressed
                    
        except Exception as e:
            self.get_logger().error(f"Failed to process image: {str(e)}")
            self.current_image = None
            
    def front_dist_callback(self, msg):
        self.front_distance = msg.data
        # self.get_logger().info(f'Recieving front distance: {self.front_distance}')
        
        # Perform classification if at right distance and haven't classified yet
        if self.current_image is not None:
            self.classify_image()
            
    def classify_image(self):
        if self.front_distance <= self.classification_distance: # and self.front_distance > self.min_distance:
            try:
                pred = predict(self.model, self.current_image)
                msg = Int32()
                msg.data = pred
                self.class_pub.publish(msg)
                self.get_logger().info(f'Published sign class: {pred}')
            except Exception as e:
                self.get_logger().error(f"Classification failed: {str(e)}")
        else:
            self.get_logger().warn(f'Not at classification distance: {self.front_distance}m')
        
def main(args=None):
    rclpy.init(args=args)
    node = SignClassifierNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()