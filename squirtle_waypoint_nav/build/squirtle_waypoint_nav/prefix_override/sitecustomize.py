import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/snaseem8/ros2_ws/src/squirtle_waypoint_nav/install/squirtle_waypoint_nav'
