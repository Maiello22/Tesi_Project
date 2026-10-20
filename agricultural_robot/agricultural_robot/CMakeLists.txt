cmake_minimum_required(VERSION 3.8)
project(agricultural_robot)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)


find_package(ros_gz_sim REQUIRED)
find_package(xacro REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(kdl_parser REQUIRED)
find_package(orocos_kdl_vendor REQUIRED)
find_package(orocos_kdl REQUIRED)
find_package(Eigen3 REQUIRED)

add_executable(clik_node src/clik_node.cpp)
target_include_directories(clik_node PRIVATE include ${EIGEN3_INCLUDE_DIR})
target_compile_features(clik_node PRIVATE cxx_std_17)
ament_target_dependencies(clik_node
  rclcpp sensor_msgs std_msgs geometry_msgs tf2_ros tf2_geometry_msgs kdl_parser orocos_kdl)

install(TARGETS clik_node DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY include/ DESTINATION include)

# Installa i file URDF/Xacro
install(DIRECTORY urdf 
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY meshes
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY config rviz
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY launch 
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY worlds 
  DESTINATION share/${PROJECT_NAME}
)

install(
  DIRECTORY models
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS scripts/aruco_detector.py scripts/inspection_node.py scripts/aruco_tf_broadcaster.py scripts/inspection_node.py
  DESTINATION lib/${PROJECT_NAME})
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()






ament_package()