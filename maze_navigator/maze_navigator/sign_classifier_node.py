import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy
from std_msgs.msg import Int32, Float32, Float32MultiArray
from cv_bridge import CvBridge
import cv2
import numpy as np
from maze_navigator.model import initialize_model, predict

class SignClassifierNode(Node):
    def __init__(self):
        super().__init__('sign_classifier_node')

        # Configure QoS for image subscription
        qos_profile = QoSProfile(
            depth=5,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            reliability=QoSReliabilityPolicy.BEST_EFFORT
        )

        # Declare display parameters
        self.declare_parameter('show_image_bool', True)
        self.declare_parameter('window_name', 'frame')
        self._display_image = self.get_parameter('show_image_bool').value
        self._window_title = self.get_parameter('window_name').value

        # Initialize OpenCV window if display is enabled
        if self._display_image:
            cv2.namedWindow(self._window_title, cv2.WINDOW_AUTOSIZE)
            cv2.moveWindow(self._window_title, 50, 50)
            cv2.setMouseCallback(self._window_title, self._click_event)

        # Load classification model
        model_path = 'knn_model_color.pkl'  # Adjust path as needed
        try:
            self._model = initialize_model(model_path)
        except Exception as e:
            self.get_logger().error(f'Failed to load model: {str(e)}')
            raise

        # Classification parameters
        self._classification_distance = 1.0
        self._min_distance = 0.2
        self._cone_angle = np.deg2rad(14.0)

        # Subscribers
        self._image_subscriber = self.create_subscription(
            CompressedImage, '/image_raw/compressed', self._image_callback, qos_profile)
        self._front_dist_subscriber = self.create_subscription(
            Float32, '/front_dist', self._front_dist_callback, 10)

        # Publishers
        self._class_publisher = self.create_publisher(Int32, '/sign_class', 10)
        self._coord_publisher = self.create_publisher(Float32MultiArray, '/coordinates', 10)

        # State variables
        self._current_image = None
        self._front_distance = float('inf')
        self._coord = None
        self._bridge = CvBridge()

    def _click_event(self, event, x, y, flags, param):
        # Log HSV values on left-click
        if event == cv2.EVENT_LBUTTONDOWN and self._current_image is not None:
            hsv_frame = cv2.cvtColor(self._current_image, cv2.COLOR_BGR2HSV)
            h, s, v = hsv_frame[y, x]
            lower_hsv = np.array([max(0, h - 10), max(0, s - 40), max(0, v - 40)])
            upper_hsv = np.array([min(179, h + 10), min(255, s + 40), min(255, v + 40)])
            self.get_logger().info(f'HSV lower: {lower_hsv}, upper: {upper_hsv}')

    def _image_callback(self, msg):
        try:
            # Convert compressed image to BGR
            self._current_image = self._bridge.compressed_imgmsg_to_cv2(msg, 'bgr8')
            height, width, _ = self._current_image.shape

            # Detect sign region
            mask, _ = self._isolate_sign_by_color(self._current_image)
            result = self._find_sign_region(self._current_image, mask)
            x = y = w = h = 0
            if result is not None:
                _, (x, y, w, h) = result

            # Calculate centroid coordinates
            centroid_x = x + w // 2
            centroid_y = y + h // 2
            self._coord = [float(centroid_x), float(centroid_y), float(width // 2), float(height // 2)] if w >= 1 and h >= 1 else [float(width // 2), float(height // 2), float(width // 2), float(height // 2)]

            # Publish coordinates
            coord_msg = Float32MultiArray()
            coord_msg.data = self._coord
            self._coord_publisher.publish(coord_msg)

            # Display image with bounding box if enabled
            if self._display_image:
                cv2.rectangle(self._current_image, (x, y), (x + w, y + h), (0, 255, 20), 3)
                cv2.imshow(self._window_title, self._current_image)
                if cv2.waitKey(1) == ord('q'):
                    cv2.destroyAllWindows()

        except Exception as e:
            self.get_logger().error(f'Image processing error: {str(e)}')
            self._current_image = None

    def _isolate_sign_by_color(self, image):
        # Define HSV color ranges for sign detection
        color_ranges = {
            'red1': ([0, 150, 130], [10, 255, 255]),
            'red2': ([160, 150, 130], [179, 255, 255]),
            'blue': ([100, 80, 60], [130, 230, 200]),
            'green': ([65, 90, 35], [95, 255, 215])
        }

        hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        best_mask = None
        best_color = None
        best_area = 0

        # Check each color range
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

        # Combine red ranges if detected
        if best_color in ['red1', 'red2']:
            mask1 = cv2.inRange(hsv_img, np.array(color_ranges['red1'][0]), np.array(color_ranges['red1'][1]))
            mask2 = cv2.inRange(hsv_img, np.array(color_ranges['red2'][0]), np.array(color_ranges['red2'][1]))
            best_mask = cv2.bitwise_or(mask1, mask2)
            best_mask = cv2.morphologyEx(best_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
            best_mask = cv2.morphologyEx(best_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
            best_color = 'red'

        return best_mask, best_color

    def _find_sign_region(self, image, mask):
        # Extract sign region with area and aspect ratio filters
        if mask is None:
            return None

        img_h, img_w = image.shape[:2]
        max_box_area = img_h * img_w * 0.4

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        if area < 150 or area > max_box_area:
            return None

        x, y, w, h = cv2.boundingRect(largest)
        margin = 20
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(img_w - x, w + 2 * margin)
        h = min(img_h - y, h + 2 * margin)

        if w * h > max_box_area:
            return None

        ar = float(w) / h if h > 0 else 0
        if ar < 0.5 or ar > 2.0:
            return None

        region = image[y:y+h, x:x+w]
        return (region, (x, y, w, h)) if region.size > 0 else None

    def _front_dist_callback(self, msg):
        # Trigger classification based on distance
        self._front_distance = msg.data
        if self._min_distance < self._front_distance <= self._classification_distance:
            self._classify_image()

    def _classify_image(self):
        # Classify the current image
        if self._current_image is not None:
            try:
                pred = predict(self._model, self._current_image)
                msg = Int32()
                msg.data = pred
                self._class_publisher.publish(msg)
                self.get_logger().info(f'Published sign class: {pred}')
            except Exception as e:
                self.get_logger().error(f'Classification error: {str(e)}')

def main():
    rclpy.init()
    node = SignClassifierNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
