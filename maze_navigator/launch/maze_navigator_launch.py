from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description(): 
    return LaunchDescription([
        Node(
            package='maze_navigator',
            executable='sign_classifier_node',
            name='sign_classifier_node',
        ),
        Node(
            package='maze_navigator',
            executable='navigator_node',
            name='navigator_node'
        )
        # Node(
        #     package='maze_navigator',
        #     executable='debug',
        #     name='debug'
        # ),
        # Node(
        #     package='maze_navigator',
        #     executable='pose_listener',
        #     name='pose_listener'
        # )
    ])