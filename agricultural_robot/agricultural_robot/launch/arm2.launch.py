import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, RegisterEventHandler, SetEnvironmentVariable
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch_ros.parameter_descriptions import ParameterValue

def generate_launch_description():
    pkg_share = FindPackageShare('agricultural_robot')
    
    # 0. Configurazione del percorso per i modelli di Gazebo
    pkg_path = get_package_share_directory('agricultural_robot')
    models_path = os.path.join(pkg_path, 'models')

    # Impostiamo le variabili d'ambiente per consentire a Gazebo di trovare "model://aruco_tag"
    if 'GZ_SIM_RESOURCE_PATH' in os.environ:
        os.environ['GZ_SIM_RESOURCE_PATH'] += os.pathsep + models_path
    else:
        os.environ['GZ_SIM_RESOURCE_PATH'] = models_path

    if 'IGN_GAZEBO_RESOURCE_PATH' in os.environ:
        os.environ['IGN_GAZEBO_RESOURCE_PATH'] += os.pathsep + models_path
    else:
        os.environ['IGN_GAZEBO_RESOURCE_PATH'] = models_path

    # 1. Caricamento del file URDF/Xacro senza argomenti di namespace o yaml
    robot_description_content = Command([
        'xacro ',
        PathJoinSubstitution([pkg_share, 'urdf', 'arm2.urdf.xacro'])
    ])
    
    robot_description = {
        'robot_description': ParameterValue(robot_description_content, value_type=str)
    }

    # 2. Bridge del Clock e delle TF per Gazebo / ROS 2
    clock_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
            '/tf@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V',
            '/tf_static@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V'
        ],
        output='screen'
    )
    
    # 3. Avvio di Gazebo Sim con il mondo specificato
    world_file = PathJoinSubstitution([pkg_share, 'worlds', 'default.sdf'])
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('ros_gz_sim'),
                'launch',
                'gz_sim.launch.py'
            ])
        ]),
        launch_arguments={
            'gz_args': [world_file, ' -r'],
        }.items()
    )

    # 4. Robot State Publisher (global, senza namespace)
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[robot_description, {'use_sim_time': True}]
    )

    # 5. Spawner del robot in Gazebo (leggendo dal topic globale /robot_description)
    spawn_entity = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-name', 'arm2',
            '-topic', '/robot_description',
            '-x', '0.0',
            '-y', '0.0',
            '-z', '0.0'
        ],
        output='screen'
    )

    # 6. Spawner dei controller ROS 2
    joint_state_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=[
            'joint_state_broadcaster',
            '--controller-manager-timeout', '60'
        ],
        output='screen'
    )

    arm_trajectory_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=[
            'arm_trajectory_controller',
            '--controller-manager-timeout', '60'
        ],
        output='screen'
    )

    trajectory_after_position = RegisterEventHandler(
        OnProcessExit(
            target_action=joint_state_spawner,
            on_exit=[arm_trajectory_spawner]
        )
    )

    bridge_camera = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/camera@sensor_msgs/msg/Image[gz.msgs.Image',
            '/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
        ],
        output='screen'
    )

    # 7. Nodo per il rilevamento di più marker ArUco (aruco_ros)
    aruco_node = Node(
        package='aruco_ros',
        executable='marker_publisher',
        name='aruco_marker_publisher',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'marker_size': 0.15,                 # Dimensione del marker in metri
            'camera_frame': 'camera_optical_link' # Frame ottico della fotocamera definito nell'URDF
        }],
        remappings=[
            ('/camera_info', '/camera_info'),
            ('/image', '/camera')
        ]
    )

    # 8. Configurazione ed avvio di RViz 2
    rviz_config_file = PathJoinSubstitution([pkg_share, 'config', 'arm2.rviz'])
    
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config_file],
        parameters=[{'use_sim_time': True}]
    )

    return LaunchDescription([
        gazebo,
        clock_bridge,
        robot_state_publisher,
        spawn_entity,
        joint_state_spawner,
        trajectory_after_position,
        bridge_camera,
        aruco_node,
        rviz_node
    ])