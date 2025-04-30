import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy
import numpy as np
import cv2
from cv_bridge import CvBridge

class ImageDebugNode(Node):
    def __init__(self):
        super().__init__('image_debug_node')

        # Declare parameters for display settings
        self.declare_parameter('show_image_bool', True)
        self.declare_parameter('window_name', 'frame')
        self._display_image = self.get_parameter('show_image_bool').value
        self._window_title = self.get_parameter('window_name').value

        # Initialize OpenCV window if display is enabled
        if self._display_image:
            cv2.namedWindow(self._window_title, cv2.WINDOW_AUTOSIZE)
            cv2.moveWindow(self._window_title, 50, 50)
            cv2.setMouseCallback(self._window_title, self._click_event)

        # Configure QoS for image subscription
        qos_profile = QoSProfile(
            depth=5,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            reliability=QoSReliabilityPolicy.BEST_EFFORT
        )

        # Subscribe to compressed image topic
        self._image_subscriber = self.create_subscription(
            CompressedImage,
            '/image_raw/compressed',
            self._image_callback,
            qos_profile
        )

        self._bridge = CvBridge()
        self._current_image = None
        self._lower_hsv = None
        self._upper_hsv = None

    def _click_event(self, event, x, y, flags, param):
        # Log HSV values when clicking on the image
        if event == cv2.EVENT_LBUTTONDOWN and self._current_image is not None:
            hsv_frame = cv2.cvtColor(self._current_image, cv2.COLOR_BGR2HSV)
            h, s, v = hsv_frame[y, x]
            self._lower_hsv = np.array([max(0, h - 10), max(0, s - 40), max(0, v - 40)])
            self._upper_hsv = np.array([min(179, h + 10), min(255, s + 40), min(255, v + 40)])
            self.get_logger().info(f'HSV lower: {self._lower_hsv}, upper: {self._upper_hsv}')

    def _image_callback(self, msg):
        try:
            # Convert compressed image to BGR
            self._current_image = self._bridge.compressed_imgmsg_to_cv2(msg, 'bgr8')
            
            # Detect sign by color and find its region
            mask, color_name = self._isolate_sign_by_color(self._current_image)
            sign_region, rect_coord = self._find_sign_region(self._current_image, mask)
            x, y, w, h = rect_coord
            mask_area = cv2.countNonZero(mask)
            self.get_logger().info(f'Mask area: {mask_area}')

            # Use cropped sign region if valid, else full image
            display_image = sign_region if sign_region is not None and mask_area > 1000 else self._current_image

            # Display image with bounding box if enabled
            if self._display_image and display_image is not None:
                cv2.rectangle(self._current_image, (x, y), (x + w, y + h), (0, 255, 20), 3)
                cv2.imshow(self._window_title, self._current_image)
                if cv2.waitKey(1) == ord('q'):
                    cv2.destroyAllWindows()
                    raise SystemExit

        except Exception as e:
            self.get_logger().error(f'Image processing failed: {str(e)}')
            self._current_image = None

    def _isolate_sign_by_color(self, image):
        # Define HSV color ranges for sign detection
        color_ranges = {
            'red1': ([0, 175, 175], [10, 255, 255]),
            'red2': ([160, 130, 140], [179, 255, 255]),
            'blue': ([102, 85, 65], [135, 215, 190]),
            'green': ([68, 104, 39], [98, 254, 212])
        }

        hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        best_mask = None
        best_color = None
        best_area = 0

        # Check each color range for the largest contour
        for color_name, (lower, upper) in color_ranges.items():
            mask = cv2.inRange(hsv_img, np.array(lower), np.array(upper))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)
                if area > best_area and area > 100:
                    best_area = area
                    best_mask = mask
                    best_color = color_name

        # Combine red ranges if red is detected
        if best_color in ['red1', 'red2']:
            mask1 = cv2.inRange(hsv_img, np.array(color_ranges['red1'][0]), np.array(color_ranges['red1'][1]))
            mask2 = cv2.inRange(hsv_img, np.array(color_ranges['red2'][0]), np.array(color_ranges['red2'][1]))
            best_mask = cv2.bitwise_or(mask1, mask2)
            best_mask = cv2.morphologyEx(best_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
            best_mask = cv2.morphologyEx(best_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
            best_color = 'red'

        return best_mask, best_color

    def _find_sign_region(self, image, mask):
        # Extract sign region from mask
        if mask is None:
            return None, (0, 0, 0, 0)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, (0, 0, 0, 0)

        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) < 100:
            return None, (0, 0, 0, 0)

        x, y, w, h = cv2.boundingRect(largest_contour)
        margin = 10
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(image.shape[1] - x, w + 2 * margin)
        h = min(image.shape[0] - y, h + 2 * margin)

        sign_region = image[y:y+h, x:x+w]
        return sign_region if sign_region.size > 0 else None, (x, y, w, h)

def main():
    rclpy.init()
    node = ImageDebugNode()
    
    try:
        rclpy.spin(node)
    except SystemExit:
        rclpy.logging.get_logger('ImageDebugNode').info('Shutting down')
    
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
