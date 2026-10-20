from launch import LaunchDescription
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    clik_params = PathJoinSubstitution(
        [FindPackageShare("agricultural_robot"), "config", "clik.yaml"]
    )
    return LaunchDescription([
        Node(
            package="agricultural_robot",
            executable="clik_node",
            name="clik_node",
            output="screen",
            parameters=[clik_params],
        )
    ])
