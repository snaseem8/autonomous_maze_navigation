#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, LaserScan
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy
from std_msgs.msg import Int32, Float32, Float32MultiArray
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
        
        # Set Parameters to show image
        self.declare_parameter('show_image_bool', True)
        self.declare_parameter('window_name', "frame")
        self.current_image = None

        # Determine Window Showing Based on Input
        self._display_image = bool(self.get_parameter('show_image_bool').value)

        # Declare some variables
        self._titleOriginal = self.get_parameter('window_name').value
        if self._display_image:
            cv2.namedWindow(self._titleOriginal, cv2.WINDOW_AUTOSIZE)
            cv2.moveWindow(self._titleOriginal, 50, 50)
            cv2.setMouseCallback(self._titleOriginal, self._click_event)  # Set mouse click event
        
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
        self.coord = None
        
        # Subscribers
        self.image_sub = self.create_subscription(
            CompressedImage, '/image_raw/compressed', self.image_callback, image_qos_profile_img)
        self.front_dist_sub = self.create_subscription(
            Float32, '/front_dist', self.front_dist_callback, 10)
        
        # Publishers
        self.class_pub = self.create_publisher(Int32, '/sign_class', 10)
        self._coord_publisher = self.create_publisher(Float32MultiArray, '/coordinates', 10)
        
        # State
        self.current_image = None
        self.front_distance = float('inf')
        
    def _click_event(self, event, x, y, flags, param):
        """ Callback function to get HSV value from a clicked pixel. """
        if event == cv2.EVENT_LBUTTONDOWN and self.img is not None:
            hsv_frame = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
            clicked_hsv = hsv_frame[y, x]  # Get HSV value at click
            h, s, v = clicked_hsv

            # Define a small range around the clicked HSV value
            self.lower_hsv = np.array([max(0, h - 10), max(0, s - 40), max(0, v - 40)])
            self.upper_hsv = np.array([min(179, h + 10), min(255, s + 40), min(255, v + 40)])
            
            # self.get_logger().info(f"HSV lower: {self.lower_hsv}.   HSV upper: {self.upper_hsv}.")
        
    def image_callback(self, CompressedImage):
        try:
            # Convert compressed image to OpenCV BGR format
            self.img = CvBridge().compressed_imgmsg_to_cv2(CompressedImage, "bgr8")
            
            # Attempt to isolate sign using color thresholding
            mask, color_name = self.isolate_sign_by_color(self.img)

            # Initialize default values
            x, y, w, h = 0, 0, 0, 0
            # Check if find_sign_region returns something before unpacking
            result = self.find_sign_region(self.img, mask)
            if result is not None:
                sign_region, rect_coord = result
                x, y, w, h = rect_coord
            
            # get pixel centroid of bounding box
            height, width, _ = self.img.shape
            centroid_x = x + w // 2
            centroid_y = y + h // 2
            if w >= 1 and h >= 1:
                # Convert x_object pixel value to degrees
                self.coord = [float(centroid_x), float(centroid_y), float(width//2), float(height//2)] #centroid x, centroid y, image center x, image center y
            else:
                self.coord = [float(width//2), float(height//2), float(width//2), float(height//2)] # if bounding box returns 0s, set centroid to image center so there is no error
            
            self.current_image = self.img

        except Exception as e:
            self.get_logger().error(f"Failed to process image: {str(e)}")
            self.current_image = None
            
        # Publish coordinates
        if self.coord is not None:
            coord_msg = Float32MultiArray()
            coord_msg.data = self.coord
            self._coord_publisher.publish(coord_msg)
            # self.get_logger().info(f"Publishing coordinates: {coord_msg.data}")    
        
        # Displaying image for debugging
        if self.current_image is not None:    
            # Display the processed frame and bounding box
            cv2.rectangle(self.img, (x, y), (x + w, y + h), (0, 255, 20), 3)
            cv2.imshow('frame', self.img)

            # Wait for key press and close window
            if cv2.waitKey(1) == ord('q'):
                cv2.destroyAllWindows()  # Close all OpenCV windows when 'q' is pressed

    def isolate_sign_by_color(self, image):
        """Identify sign regions using color thresholding"""
        
        # # Color ranges in HSV for different sign colors
        # COLOR_RANGES = {
        #     'red1': ([0, 175, 175], [10, 255, 255]),
        #     'red2': ([160, 130, 140], [179, 255, 255]),
        #     'blue': ([102, 85, 65], [135, 215, 190]),
        #     'green': ([68, 104, 39], [98, 254, 212])}

        # More focused color ranges that should still be robust
        COLOR_RANGES = {
            'red1': ([0, 150, 130], [10, 255, 255]),      # Red (first range)
            'red2': ([160, 150, 130], [179, 255, 255]),   # Red (second range)
            'blue': ([100, 80, 60], [130, 230, 200]),     # Blue
            'green': ([65, 90, 35], [95, 255, 215])       # Green
        }
        
        hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        best_mask = None
        best_color = None
        best_area = 0
        
        # Try each color range
        for color_name, (lower, upper) in COLOR_RANGES.items():
            lower = np.array(lower)
            upper = np.array(upper)
            mask = cv2.inRange(hsv_img, lower, upper)
            
            # Clean up mask
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)
                
                if area > best_area and area > 100:
                    best_area = area
                    best_mask = mask
                    best_color = color_name
        
        # Special handling for red (which spans two HSV ranges)
        if best_color in ['red1', 'red2']:
            # Combine both red masks
            red1_lower = np.array(COLOR_RANGES['red1'][0])
            red1_upper = np.array(COLOR_RANGES['red1'][1])
            red2_lower = np.array(COLOR_RANGES['red2'][0])
            red2_upper = np.array(COLOR_RANGES['red2'][1])
            
            mask1 = cv2.inRange(hsv_img, red1_lower, red1_upper)
            mask2 = cv2.inRange(hsv_img, red2_lower, red2_upper)
            best_mask = cv2.bitwise_or(mask1, mask2)
            
            kernel = np.ones((5, 5), np.uint8)
            best_mask = cv2.morphologyEx(best_mask, cv2.MORPH_OPEN, kernel)
            best_mask = cv2.morphologyEx(best_mask, cv2.MORPH_CLOSE, kernel)
            
            best_color = 'red'
        
        return best_mask, best_color

    def find_sign_region(self, image, mask):
        """Extract the sign region using the color mask with basic filtering"""
        if mask is None:
            return None
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        
        # Check both minimum AND maximum area
        min_area = 150
        max_area = image.shape[0] * image.shape[1] * 0.4  # Max 40% of frame
        
        if area < min_area or area > max_area:
            return None
        
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Aspect ratio filter - most signs are roughly square
        aspect_ratio = float(w) / h if h > 0 else 0
        if aspect_ratio < 0.5 or aspect_ratio > 2.0:
            return None
        
        # Add a small margin
        margin = 20
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(image.shape[1] - x, w + 2*margin)
        h = min(image.shape[0] - y, h + 2*margin)
        
        rect_coord = (x, y, w, h)
        
        sign_region = image[y:y+h, x:x+w]
        
        if sign_region.size == 0:
            return None
        
        return sign_region, rect_coord
            
    def front_dist_callback(self, msg):
        self.front_distance = msg.data
        # self.get_logger().info(f'Recieving front distance: {self.front_distance}')
        
        # Perform classification if at right distance and haven't classified yet
        if self.current_image is not None:
            self.classify_image()
            
    # def classify_image(self):
    #     if self.front_distance <= self.classification_distance: # and self.front_distance > self.min_distance:
    #         try:
    #             pred = predict(self.model, self.current_image)
    #             msg = Int32()
    #             msg.data = pred
    #             self.class_pub.publish(msg)
    #             self.get_logger().info(f'Published sign class: {pred}')
    #         except Exception as e:
    #             self.get_logger().error(f"Classification failed: {str(e)}")
    #     else:
    #         self.get_logger().warn(f'Not at classification distance: {self.front_distance}m')

    def classify_image(self):
        try:
            if self.current_image is not None:
                pred = predict(self.model, self.current_image)
                msg = Int32()
                msg.data = pred
                self.class_pub.publish(msg)
                self.get_logger().info(f'Published sign class: {pred}')
        except Exception as e:
            self.get_logger().error(f"Classification failed: {str(e)}")
            
        
def main(args=None):
    rclpy.init(args=args)
    node = SignClassifierNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()