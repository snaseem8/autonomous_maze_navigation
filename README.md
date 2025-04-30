<<<<<<< HEAD
# Robot Navigation and Object Tracking Repository

This repository contains ROS2 packages for robot navigation, object tracking, and maze navigation on the TurtleBot3 Burger, using a combination of computer vision, LIDAR, odometry, PID control, and SLAM. Each package builds on the previous one to achieve the end goal of navigating a maze environment by detecting road signs, avoiding obstacles, and following waypoints.

## Repository Structure

The repository is organized into six folders, each containing a specific component:

- **`object_tracker`**: A ROS2 package for rotating the robot to track an object based on the centroid of its pixel coordinates in camera images.
- **`chase_object`**: A ROS2 package for tracking and chasing an object using camera centroid data, with LIDAR used to gauge object distance and maintain a certain distance.
- **`navigate_to_goal`**: A ROS2 package for navigating to predefined goal points, incorporating obstacle avoidance using LIDAR data.
- **`waypoint_navigator`**: A ROS2 package for publishing a sequence of waypoints to the ROS2 navigation stack based on the robot's AMCL pose.
- **`maze_navigator`**: A ROS2 package for navigating a maze by detecting and interpreting road signs using a trained KNN classifier and LIDAR data.
- **`image_classifier`**: A folder containing scripts and data for training a KNN classifier to detect road signs for the `maze_navigator` package.

## Package Details

### 1. Object Tracker (`object_tracker`)
- **Purpose**: Rotates the robot to align with an object detected in camera images.
- **Functionality**:
  - Subscribes to `/image_raw/compressed` to receive camera images.
  - Processes images to detect the object's centroid using HSV thresholding.
  - Publishes angular velocity commands to `/cmd_vel` based on the angular error between the object's centroid and the camera's center.
- **Key Nodes**:
  - `debug.py`: Displays processed images for debugging.
  - `find_object.py`: Detects the object and publishes its centroid coordinates to `/coordinates`.
  - `rotate_robot.py`: Subscribes to `/coordinates` and publishes `/cmd_vel` to rotate the robot.
- **Dependencies**: `sensor_msgs`, `std_msgs`, `geometry_msgs`, `cv_bridge`, `opencv-python`, `numpy`.

### 2. Chase Object (`chase_object`)
- **Purpose**: Tracks and chases an object while maintaining a specific distance using LIDAR data.
- **Functionality**:
  - Subscribes to `/image_raw/compressed` for object detection and `/scan` for LIDAR data.
  - Publishes object coordinates to `/coordinates` and velocity commands to `/cmd_vel`.
  - Uses centroid-based rotation and LIDAR to gauge object distance, adjusting motion to maintain a desired range.
- **Key Nodes**:
  - `debug.py`: Displays processed images for debugging.
  - `find_object.py`: Detects the object and publishes its centroid coordinates.
  - `get_object_range.py`: Processes LIDAR data to estimate object distance.
  - `chase_object.py`: Controls the robot to chase the object while maintaining distance.
- **Dependencies**: `sensor_msgs`, `std_msgs`, `geometry_msgs`, `cv_bridge`, `opencv-python`, `numpy`.

### 3. Navigate to Goal (`navigate_to_goal`)
- **Purpose**: Navigates the robot to a sequence of predefined goal points with obstacle avoidance.
- **Functionality**:
  - Subscribes to `/odom` for odometry and `/object_distance` for obstacle proximity.
  - Publishes velocity commands to `/cmd_vel` to reach goals using a P-controller.
  - Switches to avoidance waypoints when obstacles are detected within 0.4 meters.
  - Pauses for 7 seconds after reaching each goal or avoidance waypoint.
- **Key Nodes**:
  - `go_to_goal.py`: Computes errors to goals and publishes velocity commands.
  - `avoid_obstacle.py`: Processes LIDAR data and publishes obstacle distances to `/object_distance`.
- **Dependencies**: `nav_msgs`, `geometry_msgs`, `std_msgs`, `numpy`.

### 4. Waypoint Navigator (`waypoint_navigator`)
- **Purpose**: Publishes a sequence of waypoints for the ROS2 navigation stack.
- **Functionality**:
  - Subscribes to `/amcl_pose` for the robot's localized pose.
  - Publishes `PoseStamped` messages to `/goal_pose` for the navigation stack.
  - Advances to the next waypoint when the robot is within 0.5 meters of the current goal, pausing for 3 seconds.
- **Key Nodes**:
  - `publish_waypoints.py`: Publishes waypoints based on AMCL pose.
- **Dependencies**: `geometry_msgs`, `numpy`.

### 5. Maze Navigator (`maze_navigator`)
- **Purpose**: Navigates a maze by interpreting road signs and avoiding walls.
- **Functionality**:
  - Subscribes to `/image_raw/compressed` for camera images, `/scan` for LIDAR, `/amcl_pose` for localization, and `/coordinates` for sign centroids.
  - Uses a trained KNN classifier to detect road signs (e.g., left, right, stop, goal).
  - Publishes sign classifications to `/sign_class`, front distances to `/front_dist`, and velocity commands to `/cmd_vel`.
  - Implements a state machine for turning, aligning with walls, and moving forward based on sign instructions.
- **Key Nodes**:
  - `debug.py`: Displays processed images with bounding boxes for debugging.
  - `sign_classifier_node.py`: Detects and classifies road signs using color segmentation and KNN.
  - `navigator_node.py`: Controls robot navigation based on sign classifications and LIDAR data.
  - `pose_listener.py`: Listens to AMCL pose and logs the robot's position.
- **Dependencies**: `sensor_msgs`, `geometry_msgs`, `std_msgs`, `cv_bridge`, `opencv-python`, `numpy`, `transforms3d`, `tf2_ros`.

### 6. Image Classifier (`image_classifier`)
- **Purpose**: Contains scripts and data for training a KNN classifier for road sign detection.
- **Functionality**:
  - Includes training scripts to generate a KNN model (`knn_model_color.pkl`) based on image data.
  - The trained model is used by the `sign_classifier_node.py` in the `maze_navigator` package to classify road signs.
- **Contents**:
  - Scripts for data preprocessing, feature extraction, and model training.
  - Sample image datasets for road signs (e.g., left, right, stop, goal).
- **Dependencies**: `opencv-python`, `numpy`, `scikit-learn`.
