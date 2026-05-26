cmake_minimum_required(VERSION 3.0.2)
project(city_query)

find_package(catkin REQUIRED COMPONENTS
  message_generation
  rospy
  std_msgs
)

add_service_files(
  FILES
  CityQuery.srv
)

generate_messages(
  DEPENDENCIES
  std_msgs
)

catkin_package(
  CATKIN_DEPENDS message_runtime rospy std_msgs
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)