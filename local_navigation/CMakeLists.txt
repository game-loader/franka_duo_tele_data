cmake_minimum_required(VERSION 3.8)
project(tmr_local_navigation)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

ament_python_install_package(${PROJECT_NAME})
install(PROGRAMS
  ${PROJECT_NAME}/odom_frame_adapter.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME odom_frame_adapter
)
install(PROGRAMS
  ${PROJECT_NAME}/dual_laser_merger.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME dual_laser_merger
)
install(DIRECTORY launch resource config
  DESTINATION share/${PROJECT_NAME}
)
install(FILES package.xml setup.py setup.cfg
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
