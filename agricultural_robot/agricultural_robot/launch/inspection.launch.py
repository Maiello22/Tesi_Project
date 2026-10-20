from launch import LaunchDescription
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    params = PathJoinSubstitution(
        [FindPackageShare("agricultural_robot"), "config", "inspection.yaml"]
    )
    return LaunchDescription([
        Node(
            package="agricultural_robot",
            executable="inspection_node.py",
            name="inspection_node",
            output="screen",
            parameters=[params],
        )
    ])
