from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='squirtle_object_follower',
            namespace='find_object',
            executable='find_object',
            name='find_object'
        ),
        Node(
            package='squirtle_object_follower',
            namespace='',
            executable='',
            name=''
        ),
    ])