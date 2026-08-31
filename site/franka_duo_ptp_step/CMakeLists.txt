cmake_minimum_required(VERSION 3.16)
project(franka_duo_ptp_step)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(ament_cmake REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(franka_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(moveit_core REQUIRED)
find_package(moveit_ros_planning REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(trajectory_msgs REQUIRED)

add_executable(left_ptp_step
  src/left_ptp_step.cpp
)
add_executable(duo_ptp_episode
  src/duo_ptp_episode.cpp
)
add_executable(jtc_command_relay
  src/jtc_command_relay.cpp
)
ament_target_dependencies(left_ptp_step
  franka_msgs
  geometry_msgs
  moveit_core
  moveit_ros_planning
  rclcpp
  rclcpp_action
  sensor_msgs
)
ament_target_dependencies(duo_ptp_episode
  franka_msgs
  moveit_core
  moveit_ros_planning
  rclcpp
  rclcpp_action
  sensor_msgs
  builtin_interfaces
  trajectory_msgs
)
ament_target_dependencies(jtc_command_relay
  rclcpp
  trajectory_msgs
)

install(
  TARGETS
    left_ptp_step
    duo_ptp_episode
    jtc_command_relay
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)
install(
  DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
  FILES_MATCHING PATTERN "*.launch.py"
)
install(
  DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
  FILES_MATCHING PATTERN "*.yaml"
)
install(
  FILES README.md
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
