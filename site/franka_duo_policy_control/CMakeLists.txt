cmake_minimum_required(VERSION 3.16)
project(franka_duo_policy_control)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(ament_cmake REQUIRED)
find_package(controller_interface REQUIRED)
find_package(eigen3_cmake_module REQUIRED)
find_package(franka_semantic_components REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(hardware_interface REQUIRED)
find_package(pluginlib REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(realtime_tools REQUIRED)
find_package(std_msgs REQUIRED)
find_package(Eigen3 REQUIRED)

add_library(policy_cartesian_controllers SHARED
  src/policy_cartesian_pose_controller.cpp
)
target_include_directories(policy_cartesian_controllers PUBLIC
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
  "$<INSTALL_INTERFACE:include>"
)
target_link_libraries(policy_cartesian_controllers Eigen3::Eigen)
ament_target_dependencies(policy_cartesian_controllers
  controller_interface
  franka_semantic_components
  geometry_msgs
  hardware_interface
  pluginlib
  rclcpp
  rclcpp_lifecycle
  realtime_tools
)

add_executable(policy_action_relay
  src/policy_action_relay.cpp
)
target_include_directories(policy_action_relay PRIVATE
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
)
target_link_libraries(policy_action_relay Eigen3::Eigen)
ament_target_dependencies(policy_action_relay
  geometry_msgs
  rclcpp
  std_msgs
)

pluginlib_export_plugin_description_file(controller_interface franka_duo_policy_control.xml)

install(
  DIRECTORY include/
  DESTINATION include
)
install(
  TARGETS
    policy_cartesian_controllers
    policy_action_relay
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)
install(
  DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
  FILES_MATCHING PATTERN "*.yaml"
)
install(
  DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
  FILES_MATCHING PATTERN "*.launch.py"
)
install(
  FILES README.md franka_duo_policy_control.xml
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_action_math
    test/test_action_math.cpp
  )
  target_include_directories(test_action_math PRIVATE include)
  target_link_libraries(test_action_math Eigen3::Eigen)
endif()

ament_package()
