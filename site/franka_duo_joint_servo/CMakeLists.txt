cmake_minimum_required(VERSION 3.16)
project(franka_duo_joint_servo)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(ament_cmake REQUIRED)
find_package(eigen3_cmake_module REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(moveit_core REQUIRED)
find_package(moveit_ros_planning REQUIRED)
find_package(rclcpp REQUIRED)
find_package(ruckig REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)

add_executable(policy_chunk_joint_servo src/policy_chunk_joint_servo.cpp)
target_include_directories(policy_chunk_joint_servo PRIVATE
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
)
target_link_libraries(policy_chunk_joint_servo Eigen3::Eigen ruckig::ruckig)
ament_target_dependencies(policy_chunk_joint_servo
  moveit_core
  moveit_ros_planning
  rclcpp
  sensor_msgs
  std_msgs
)

add_executable(gello_target_relay src/gello_target_relay.cpp)
ament_target_dependencies(gello_target_relay
  rclcpp
  sensor_msgs
  std_msgs
)

install(
  TARGETS policy_chunk_joint_servo gello_target_relay
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)
install(DIRECTORY include/ DESTINATION include)
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
install(FILES README.md DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_joint_timeline test/test_joint_timeline.cpp)
  target_include_directories(test_joint_timeline PRIVATE include)
  ament_add_gtest(test_joint_ruckig_tracker test/test_joint_ruckig_tracker.cpp)
  target_include_directories(test_joint_ruckig_tracker PRIVATE include)
  target_link_libraries(test_joint_ruckig_tracker ruckig::ruckig)
endif()

ament_package()
