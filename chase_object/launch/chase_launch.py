from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='squirtle_chase_object',
            executable='find_object',
            name='find_object'
        ),
        Node(
            package='squirtle_chase_object',
            executable='get_object_range',
            name='get_object_range'
        ),
        Node(
            package='squirtle_chase_object',
            executable='chase_object',
            name='chase_object'
        )
    ])