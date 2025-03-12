from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='squirtle_navigate_to_goal',
            executable='goToGoal',
            name='goToGoal'
        ),
        Node(
            package='squirtle_navigate_to_goal',
            executable='avoid_obstacle',
            name='avoid_obstacle'
        ),
    ])