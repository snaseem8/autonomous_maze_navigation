# Bare Bones Code to View the Image Published from the Turtlebot3 on a Remote Computer
# Intro to Robotics Research 7785
# Georgia Institute of Technology
# Sean Wilson, 2022

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy

import sys

import numpy as np
import cv2
from cv_bridge import CvBridge

class MinimalVideoSubscriber(Node):

    def __init__(self):		
        # Creates the node.
        super().__init__('minimal_video_subscriber')

        # Set Parameters
        self.declare_parameter('show_image_bool', True)
        self.declare_parameter('window_name', "frame")

        # Determine Window Showing Based on Input
        self._display_image = bool(self.get_parameter('show_image_bool').value)

        # Declare some variables
        self._titleOriginal = self.get_parameter('window_name').value  # Image Window Title	
        if self._display_image:
            # Set Up Image Viewing
            cv2.namedWindow(self._titleOriginal, cv2.WINDOW_AUTOSIZE)  # Viewing Window
            cv2.moveWindow(self._titleOriginal, 50, 50)  # Viewing Window Original Location
    
        # Set up QoS Profiles for passing images over WiFi
        image_qos_profile = QoSProfile(depth=5)
        image_qos_profile.history = QoSHistoryPolicy.KEEP_LAST
        image_qos_profile.durability = QoSDurabilityPolicy.VOLATILE 
        image_qos_profile.reliability = QoSReliabilityPolicy.BEST_EFFORT 

        # Declare that the minimal_video_subscriber node is subscribing to the /camera/image/compressed topic.
        self._video_subscriber = self.create_subscription(
            CompressedImage,
            '/image_raw/compressed',
            self._image_callback,
            image_qos_profile)
        self._video_subscriber  # Prevents unused variable warning.

    def _click_event(self, event, x, y, flags, param):
        """ Callback function to get HSV value from a clicked pixel. """
        if event == cv2.EVENT_LBUTTONDOWN and self.img is not None:
            hsv_frame = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
            clicked_hsv = hsv_frame[y, x]  # Get HSV value at click
            h, s, v = clicked_hsv

            # Define a small range around the clicked HSV value
            self.lower_hsv = np.array([max(0, h - 10), max(0, s - 40), max(0, v - 40)])
            self.upper_hsv = np.array([min(179, h + 10), min(255, s + 40), min(255, v + 40)])
            
            self.get_logger().info(f"HSV lower: {self.lower_hsv}.   HSV upper: {self.upper_hsv}.")
        
    def _image_callback(self, CompressedImage):
        try:
            # Convert compressed image to OpenCV BGR format
            self.img = CvBridge().compressed_imgmsg_to_cv2(CompressedImage, "bgr8")
            
            # Attempt to isolate sign using color thresholding
            mask, color_name = self.isolate_sign_by_color(self.img)
            sign_region, rect_coord = self.find_sign_region(self.img, mask)
            x, y, w, h = rect_coord
            
            mask_area = cv2.countNonZero(mask)
            self.get_logger().info(f"Mask area: {mask_area}")
            if sign_region is not None and mask_area > 1000:
                self.current_image = sign_region
                # self.get_logger().info(f"using cropped image by color ({color_name})")
            else:
                # self.get_logger().warn("Color-based sign detection failed, using full image")
                self.current_image = self.img

        except Exception as e:
            self.get_logger().error(f"Failed to process image: {str(e)}")
            self.current_image = None
        
        # Displaying image for debugging
        if self.current_image is not None:    
            # Display the processed frame with bounding box
            cv2.rectangle(self.img, (x, y), (x + w, y + h), (0, 255, 20), 3)
            cv2.imshow('frame', self.img)

            # Wait for key press and close window
            if cv2.waitKey(1) == ord('q'):
                cv2.destroyAllWindows()  # Close all OpenCV windows when 'q' is pressed

    def isolate_sign_by_color(self, image):
        """Identify sign regions using color thresholding"""
        
        # Color ranges in HSV for different sign colors
        COLOR_RANGES = {
            'red1': ([0, 175, 175], [10, 255, 255]),
            'red2': ([160, 130, 140], [179, 255, 255]),
            'blue': ([102, 85, 65], [135, 215, 190]),
            'green': ([68, 104, 39], [98, 254, 212])}
        
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
        """Extract the sign region using the color mask"""
        if mask is None:
            return None
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        
        if area < 100:
            return None
        
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Add a small margin
        margin = 10
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(image.shape[1] - x, w + 2*margin)
        h = min(image.shape[0] - y, h + 2*margin)
        
        rect_coord = (x, y, w, h)
        
        sign_region = image[y:y+h, x:x+w]
        
        if sign_region.size == 0:
            return None
        
        return sign_region, rect_coord
                

    def get_image(self):
        return self._imgBGR

    def get_user_input(self):
        return self._user_input

    def show_image(self, img):
        cv2.imshow(self._titleOriginal, img)
        # Cause a slight delay so image is displayed
        self._user_input = cv2.waitKey(50)  # Use OpenCV keystroke grabber for delay. Done
        if self.get_user_input() == ord('q'):
            cv2.destroyAllWindows()
            raise SystemExit


def main():
    rclpy.init()  # init routine needed for ROS2.
    video_subscriber = MinimalVideoSubscriber()  # Create class object to be used.
    
    try:
        rclpy.spin(video_subscriber)  # Trigger callback processing.		
    except SystemExit:
        rclpy.logging.get_logger("Camera Viewer Node Info...").info("Shutting Down")
    # Clean up and shutdown.
    video_subscriber.destroy_node()  
    rclpy.shutdown()


if __name__ == '__main__':
    main()
